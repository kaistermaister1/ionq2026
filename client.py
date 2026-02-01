"""
GameClient - Player interface for the quantum networking game server.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import requests
from qiskit import QuantumCircuit, qasm3


class GameClient:
    """Client for interacting with the game server API."""

    def __init__(self, base_url: str = "https://demo-entanglement-distillation-qfhvrahfcq-uc.a.run.app", api_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.player_id: Optional[str] = None
        self.name: Optional[str] = None
        self._cached_graph: Optional[Dict[str, Any]] = None
        # Optimistic local connectivity: if a claim succeeds but the next /status
        # response lags briefly, we still treat the edge as owned immediately.
        # Stored as canonical undirected edges (a,b) with a<=b.
        self._optimistic_owned_edges: Set[Tuple[str, str]] = set()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _get(self, path: str) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.base_url}{path}", headers=self._headers(), timeout=120)
        except requests.RequestException as e:
            # Keep callers resilient; surface minimal info.
            return {"_error": {"code": "NETWORK_ERROR", "message": str(e)}}

        try:
            payload = r.json()
        except Exception:
            payload = {}

        if not r.ok:
            # Most endpoints return {"ok": false, "error": {...}} even on 4xx.
            err = payload.get("error") or {"code": f"HTTP_{r.status_code}", "message": (r.text or "").strip()[:500]}
            return {"_error": err}

        # Successful GET endpoints are usually {"data": {...}}
        return payload.get("data", {}) or {}

    def _post(self, path: str, payload: Dict[str, Any], require_auth: bool = True) -> Dict[str, Any]:
        if require_auth and not self.api_token:
            return {"ok": False, "error": {"code": "NO_TOKEN", "message": "No API token. Register first."}}
        try:
            r = requests.post(f"{self.base_url}{path}", json=payload, headers=self._headers(), timeout=30)
        except requests.RequestException as e:
            return {"ok": False, "error": {"code": "NETWORK_ERROR", "message": str(e)}}

        try:
            resp = r.json()
        except Exception:
            resp = {}

        # For 4xx/5xx, return JSON error instead of raising, so callers can print
        # the actual server error code/message (avoids "random 400" mystery).
        if not r.ok:
            if isinstance(resp, dict) and "ok" in resp:
                return resp
            return {
                "ok": False,
                "error": resp.get("error") if isinstance(resp, dict) else None
                or {"code": f"HTTP_{r.status_code}", "message": (r.text or "").strip()[:500]},
            }

        return resp

    # ---- Core API Methods ----

    def register(self, player_id: str, name: str, location: str = "remote") -> Dict[str, Any]:
        """Register a new player. Location: "in_person" (Americas) or "remote" (AfroEuroAsia)."""
        resp = self._post("/v1/register", {"player_id": player_id, "name": name, "location": location}, require_auth=False)
        if resp.get("ok"):
            self.player_id = player_id
            self.name = name
            if "data" in resp and "api_token" in resp["data"]:
                self.api_token = resp["data"]["api_token"]
        elif resp.get("error", {}).get("code") == "PLAYER_EXISTS":
            self.player_id = player_id
            self.name = name
        return resp

    def select_starting_node(self, node_id: str) -> Dict[str, Any]:
        """Select a starting node from the candidates provided at registration."""
        if not self.player_id:
            return {"ok": False, "error": {"code": "NOT_REGISTERED", "message": "Not registered"}}
        return self._post("/v1/select_starting_node", {"player_id": self.player_id, "node_id": node_id})

    def restart(self) -> Dict[str, Any]:
        """Reset game progress (keeps player, resets starting node)."""
        if not self.player_id:
            return {"ok": False, "error": {"code": "NOT_REGISTERED", "message": "Not registered"}}
        self._optimistic_owned_edges.clear()
        return self._post("/v1/restart", {"player_id": self.player_id})

    def get_status(self, include_optimistic: bool = False) -> Dict[str, Any]:
        """
        Get current player status including score, budget, owned nodes/edges.

        Args:
            include_optimistic: If True, inject locally-tracked successful edges that the server
                hasn't reflected yet. This can make UIs feel more responsive, but can also cause
                confusion when the server still rejects follow-up actions (e.g. EDGE_NOT_ADJACENT).
                Default is False for robustness.
        """
        if not self.player_id:
            return {}
        data = self._get(f"/v1/status/{self.player_id}") or {}

        # Reconcile optimistic edges against server truth.
        server_edges = self._edges_from_status(data)
        if self._optimistic_owned_edges:
            # Drop any optimistic edges that the server now confirms.
            self._optimistic_owned_edges = {e for e in self._optimistic_owned_edges if e not in server_edges}

        # If server status is lagging, inject optimistic edges so downstream logic
        # (claimable edges, reachable nodes, interactive viz) updates immediately.
        if include_optimistic and self._optimistic_owned_edges:
            owned_edges = list(data.get("owned_edges", []) or [])
            existing = server_edges
            for (a, b) in sorted(self._optimistic_owned_edges):
                if (a, b) in existing:
                    continue
                owned_edges.append([a, b])
                existing.add((a, b))
            data["owned_edges"] = owned_edges

        return data

    def get_graph(self) -> Dict[str, Any]:
        """Get the quantum network graph structure."""
        return self._get("/v1/graph")

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get the current leaderboard."""
        return self._get("/v1/leaderboard")

    def claim_edge(
        self,
        edge: Tuple[str, str],
        circuit: QuantumCircuit,
        flag_bit: int,
        num_bell_pairs: int,
    ) -> Dict[str, Any]:
        """
        Claim an edge by submitting a distillation circuit.

        Args:
            edge: Tuple of (node_a, node_b)
            circuit: 2N-qubit QuantumCircuit with LOCC operations
            flag_bit: Classical bit index for post-selection (0 = success)
            num_bell_pairs: Number of raw Bell pairs (1-8)

        Returns:
            Response with fidelity, success_probability, threshold, and success status.
        """
        if not self.player_id:
            return {"ok": False, "error": {"code": "NOT_REGISTERED", "message": "Not registered"}}

        # IMPORTANT: The server expects the edge direction to be owned/reachable -> unowned/unreachable.
        # Some helpers (or the graph's stored ordering) may provide the reverse direction, which can
        # trigger EDGE_NOT_ADJACENT. Re-orient best-effort using your current status.
        try:
            # Use server-truth status for orientation so we don't attempt expansion too early.
            status = self.get_status(include_optimistic=False) or {}
            owned_nodes = set(status.get("owned_nodes", []) or [])
            reachable = self.get_reachable_nodes(status)
            a, b = edge
            # Prefer orienting by owned_nodes (if server uses that notion), else fallback to reachability.
            if a not in owned_nodes and b in owned_nodes:
                edge = (b, a)
            elif a not in reachable and b in reachable:
                edge = (b, a)
        except Exception:
            # Never let re-orientation break claims
            pass

        payload = {
            "player_id": self.player_id,
            "edge": [edge[0], edge[1]],
            "num_bell_pairs": int(num_bell_pairs),
            "circuit_qasm": qasm3.dumps(circuit),
            "flag_bit": int(flag_bit),
        }
        resp = self._post("/v1/claim_edge", payload)

        # Optimistic edge ownership: if the claim succeeded, treat it as owned immediately
        # even if /status hasn't reflected it yet (prevents "attack twice" UX).
        try:
            if resp.get("ok") and resp.get("data", {}).get("success"):
                self._optimistic_owned_edges.add(self._normalize_edge(edge[0], edge[1]))
        except Exception:
            # Never let bookkeeping break the API call
            pass
        return resp

    # ---- Convenience Methods ----

    def get_cached_graph(self, force: bool = False) -> Dict[str, Any]:
        """Get graph with caching (graph doesn't change during game)."""
        if force or self._cached_graph is None:
            self._cached_graph = self.get_graph()
        return self._cached_graph

    @staticmethod
    def _normalize_edge(a: str, b: str) -> Tuple[str, str]:
        """Canonical undirected edge representation for set membership."""
        return (a, b) if a <= b else (b, a)

    def _edges_from_status(self, status: Dict[str, Any]) -> Set[Tuple[str, str]]:
        edges: Set[Tuple[str, str]] = set()
        for e in status.get("owned_edges", []) or []:
            try:
                a, b = e
            except Exception:
                continue
            edges.add(self._normalize_edge(a, b))
        return edges

    def get_reachable_nodes(self, status: Optional[Dict[str, Any]] = None) -> Set[str]:
        """
        Nodes you can expand from based on connectivity, NOT vertex ownership.

        In this game:
        - `owned_edges` define your connectivity / where you can expand next.
        - `owned_nodes` reflects vertex reward winners and should not gate movement.
        """
        status = status or self.get_status() or {}
        start = status.get("starting_node")
        if not start:
            # Fallback: if server didn't provide starting_node, best-effort seed from owned_nodes
            owned_nodes = status.get("owned_nodes", []) or []
            start = owned_nodes[0] if owned_nodes else None
        if not start:
            return set()

        # Build adjacency from owned edges (undirected)
        adj: Dict[str, Set[str]] = {}
        for e in status.get("owned_edges", []) or []:
            try:
                a, b = e
            except Exception:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        # BFS/DFS from start
        seen: Set[str] = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v in adj.get(u, set()):
                if v not in seen:
                    stack.append(v)
        return seen

    def get_claimable_edges(self, include_optimistic: bool = False) -> List[Dict[str, Any]]:
        """
        Get edges adjacent to your reachable component that can be claimed.

        IMPORTANT:
        - The server enforces adjacency using your current "owned nodes" set (as returned by /status).
          In practice, you can only submit claims on edges with exactly one endpoint in `owned_nodes`.
        - We intentionally do NOT filter out edges that are already in `owned_edges`, because players
          may want to re-claim/reinforce the same edge to increase vertex claim strength.
        """
        status = self.get_status(include_optimistic=include_optimistic)
        owned = set(status.get("owned_nodes", []) or [])
        if not owned:
            return []

        graph = self.get_cached_graph()
        claimable = []
        for edge in graph.get('edges', []):
            n1, n2 = edge['edge_id']
            if (n1 in owned) != (n2 in owned):
                claimable.append(edge)
        return claimable

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific node."""
        graph = self.get_cached_graph()
        for node in graph.get('nodes', []):
            if node['node_id'] == node_id:
                return node
        return None

    def get_edge_info(self, node_a: str, node_b: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific edge."""
        graph = self.get_cached_graph()
        edge_id = tuple(sorted([node_a, node_b]))
        for edge in graph.get('edges', []):
            if tuple(sorted(edge['edge_id'])) == edge_id:
                return edge
        return None

    def print_status(self):
        """Print a formatted summary of player status."""
        status = self.get_status()
        if not status:
            print("Not registered or no status available.")
            return

        print("=" * 50)
        print(f"Player: {status.get('player_id', 'Unknown')} ({status.get('name', '')})")
        print(f"Score: {status.get('score', 0)} | Budget: {status.get('budget', 0)} bell pairs")
        print(f"Active: {'Yes' if status.get('is_active', False) else 'No'}")
        print(f"Starting node: {status.get('starting_node', 'Not selected')}")

        owned_nodes = status.get('owned_nodes', [])
        owned_edges = status.get('owned_edges', [])
        print(f"Owned: {len(owned_nodes)} nodes, {len(owned_edges)} edges")

        claimable = self.get_claimable_edges()
        print(f"Claimable edges: {len(claimable)}")
        for edge in claimable[:3]:
            print(f"  - {edge['edge_id']}: threshold={edge['base_threshold']:.2f}, difficulty={edge['difficulty_rating']}")
        if len(claimable) > 3:
            print(f"  ... and {len(claimable) - 3} more")
        print("=" * 50)
