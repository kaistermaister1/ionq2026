"""
Client - Player interface for interacting with the remote game server (HTTP/JSON).
"""

from __future__ import annotations

from os import getenv
from typing import Any, Dict, List, Optional, Set, Tuple
import requests

from qiskit import QuantumCircuit, qasm3


class GameClient:
    def __init__(self, base_url: str = "https://demo-entanglement-distillation-qfhvrahfcq-uc.a.run.app", api_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.player_id: Optional[str] = None
        self.name: Optional[str] = None
        self._cached_graph: Optional[Dict[str, Any]] = None
        self._cached_status: Optional[Dict[str, Any]] = None

    # ---- internal helpers ----
    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ionq/iQuHack2026/0.1.0"
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _get(self, path: str, require_auth: bool = False) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}{path}", headers=self._headers(), timeout=120)
        r.raise_for_status()
        return r.json().get("data", {})

    def _post(self, path: str, payload: Dict[str, Any], require_auth: bool = True) -> Dict[str, Any]:
        if require_auth and not self.api_token:
            return {"ok": False, "error": {"code": "NO_TOKEN", "message": "No API token available. Please register first."}}
        r = requests.post(f"{self.base_url}{path}", json=payload, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    # ---- API methods ----
    def register(self, player_id: str, name: str, location: str = "remote") -> Dict[str, Any]:
        """
        Register a new player.

        Args:
            player_id: Unique player identifier
            name: Player display name
            location: Either "in_person" (Americas) or "remote" (AfroEuroAsia) - defaults to "remote"

        Returns:
            Response dict with api_token, starting_candidates, etc.
        """
        resp = self._post("/v1/register", {"player_id": player_id, "name": name, "location": location}, require_auth=False)
        if resp.get("ok"):
            self.player_id = player_id
            self.name = name
            # Store the api_token returned from registration
            if "data" in resp and "api_token" in resp["data"]:
                self.api_token = resp["data"]["api_token"]
        elif resp.get("error", {}).get("code") == "PLAYER_EXISTS":
            # Player already exists, but we don't have their token
            self.player_id = player_id
            self.name = name
        return resp

    def select_starting_node(self, node_id: str) -> Dict[str, Any]:
        if not self.player_id:
            return {"ok": False, "error": {"code": "NOT_REGISTERED", "message": "Not registered"}}
        return self._post("/v1/select_starting_node", {"player_id": self.player_id, "node_id": node_id})

    def restart(self) -> Dict[str, Any]:
        if not self.player_id:
            return {"ok": False, "error": {"code": "NOT_REGISTERED", "message": "Not registered"}}
        return self._post("/v1/restart", {"player_id": self.player_id})

    def get_status(self) -> Dict[str, Any]:
        if not self.player_id:
            return {"ok": False, "error": {"code": "NOT_REGISTERED", "message": "Not registered"}}
        return self._get(f"/v1/status/{self.player_id}")

    def get_graph(self) -> Dict[str, Any]:
        return self._get("/v1/graph")

    def get_leaderboard(self) -> Dict[str, Any]:
        return self._get("/v1/leaderboard")

    def claim_edge(
        self,
        edge: Tuple[str, str],
        circuit_a: QuantumCircuit,
        circuit_b: QuantumCircuit,
        num_bell_pairs: int,
    ) -> Dict[str, Any]:
        if not self.player_id:
            return {"ok": False, "error": {"code": "NOT_REGISTERED", "message": "Not registered"}}

        payload = {
            "player_id": self.player_id,
            "edge": [edge[0], edge[1]],
            "num_bell_pairs": int(num_bell_pairs),
            "circuit_a_qasm": qasm3.dumps(circuit_a),
            "circuit_b_qasm": qasm3.dumps(circuit_b),
        }
        result = self._post("/v1/claim_edge", payload)
        # Invalidate cached status after claiming
        self._cached_status = None
        return result

    # ---- Convenience methods ----

    def refresh_status(self, force: bool = False) -> Dict[str, Any]:
        """
        Get player status with caching.

        Args:
            force: If True, bypass cache and fetch fresh data

        Returns:
            Status dict with score, budget, owned_nodes, owned_edges, etc.
        """
        if force or self._cached_status is None:
            self._cached_status = self.get_status()
        return self._cached_status

    def get_cached_graph(self, force: bool = False) -> Dict[str, Any]:
        """
        Get graph structure with caching (graph doesn't change during game).

        Args:
            force: If True, bypass cache and fetch fresh data

        Returns:
            Graph dict with nodes and edges
        """
        if force or self._cached_graph is None:
            self._cached_graph = self.get_graph()
        return self._cached_graph

    def get_owned_nodes(self, refresh: bool = False) -> List[str]:
        """
        Get list of nodes owned by the player.

        Args:
            refresh: If True, fetch fresh status from server

        Returns:
            List of node IDs
        """
        status = self.refresh_status(force=refresh)
        return status.get('owned_nodes', [])

    def get_owned_edges(self, refresh: bool = False) -> List[List[str]]:
        """
        Get list of edges owned by the player.

        Args:
            refresh: If True, fetch fresh status from server

        Returns:
            List of [node_a, node_b] pairs
        """
        status = self.refresh_status(force=refresh)
        return status.get('owned_edges', [])

    def get_claimable_edges(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of edges that can be claimed (adjacent to owned subgraph).

        Args:
            refresh: If True, fetch fresh status from server

        Returns:
            List of edge dicts with all properties (threshold, difficulty, etc.)
        """
        owned = set(self.get_owned_nodes(refresh=refresh))
        if not owned:
            return []

        graph = self.get_cached_graph()
        claimable = []
        for edge in graph.get('edges', []):
            n1, n2 = edge['edge_id']
            # Edge is claimable if one end is owned and other is not
            if (n1 in owned) != (n2 in owned):
                claimable.append(edge)

        return claimable

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific node.

        Args:
            node_id: ID of the node

        Returns:
            Node dict with utility_qubits, bonus_bell_pairs, etc., or None if not found
        """
        graph = self.get_cached_graph()
        for node in graph.get('nodes', []):
            if node['node_id'] == node_id:
                return node
        return None

    def get_edge_info(self, node_a: str, node_b: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific edge.

        Args:
            node_a: First node ID
            node_b: Second node ID

        Returns:
            Edge dict with threshold, difficulty, etc., or None if not found
        """
        graph = self.get_cached_graph()
        # Normalize edge (edges are stored in sorted order)
        edge_id = tuple(sorted([node_a, node_b]))
        for edge in graph.get('edges', []):
            if tuple(sorted(edge['edge_id'])) == edge_id:
                return edge
        return None

    def get_budget(self, refresh: bool = False) -> int:
        """
        Get current bell pair budget.

        Args:
            refresh: If True, fetch fresh status from server

        Returns:
            Current budget
        """
        status = self.refresh_status(force=refresh)
        return status.get('budget', 0)

    def get_score(self, refresh: bool = False) -> int:
        """
        Get current score.

        Args:
            refresh: If True, fetch fresh status from server

        Returns:
            Current score (sum of utility qubits from owned nodes)
        """
        status = self.refresh_status(force=refresh)
        return status.get('score', 0)

    def is_active(self, refresh: bool = False) -> bool:
        """
        Check if player is still active in the game.

        Args:
            refresh: If True, fetch fresh status from server

        Returns:
            True if player is active, False if eliminated
        """
        status = self.refresh_status(force=refresh)
        return status.get('is_active', True)

    def auto_select_best_starting_node(self, candidates: List[Dict[str, Any]],
                                       prefer: str = "utility") -> Dict[str, Any]:
        """
        Automatically select the best starting node from candidates.

        Args:
            candidates: List of candidate nodes from registration response
            prefer: Selection strategy - "utility" (max qubits), "bonus" (max bell pairs),
                   or "balanced" (sum of both)

        Returns:
            Result from select_starting_node()
        """
        if not candidates:
            return {"ok": False, "error": {"code": "NO_CANDIDATES", "message": "No candidates available"}}

        if prefer == "utility":
            best = max(candidates, key=lambda n: n['utility_qubits'])
        elif prefer == "bonus":
            best = max(candidates, key=lambda n: n['bonus_bell_pairs'])
        else:  # balanced
            best = max(candidates, key=lambda n: n['utility_qubits'] + n['bonus_bell_pairs'])

        return self.select_starting_node(best['node_id'])

    def print_status(self, refresh: bool = True):
        """
        Print a formatted summary of player status.

        Args:
            refresh: If True, fetch fresh status from server
        """
        status = self.refresh_status(force=refresh)

        print("=" * 60)
        print(f"PLAYER STATUS: {status.get('player_id', 'Unknown')}")
        print("=" * 60)
        print(f"Name:           {status.get('name', 'Unknown')}")
        print(f"Score:          {status.get('score', 0)} points")
        print(f"Budget:         {status.get('budget', 0)} bell pairs")
        print(f"Active:         {'✅ Yes' if status.get('is_active', False) else '❌ No (eliminated)'}")
        print(f"Starting node:  {status.get('starting_node', 'Not selected')}")

        owned_nodes = status.get('owned_nodes', [])
        owned_edges = status.get('owned_edges', [])
        print(f"\nOwned nodes:    {len(owned_nodes)}")
        if owned_nodes:
            for node_id in owned_nodes:
                node_info = self.get_node_info(node_id)
                if node_info:
                    print(f"  - {node_id}: {node_info['utility_qubits']} qubits, +{node_info['bonus_bell_pairs']} bonus")

        print(f"\nOwned edges:    {len(owned_edges)}")
        if owned_edges:
            for edge in owned_edges[:5]:  # Show first 5
                print(f"  - {edge}")
            if len(owned_edges) > 5:
                print(f"  ... and {len(owned_edges) - 5} more")

        claimable = self.get_claimable_edges()
        print(f"\nClaimable edges: {len(claimable)}")
        if claimable:
            for edge in claimable[:5]:  # Show first 5
                print(f"  - {edge['edge_id']}: threshold={edge['base_threshold']:.2f}, difficulty={edge['difficulty_rating']}")
            if len(claimable) > 5:
                print(f"  ... and {len(claimable) - 5} more")
        print("=" * 60)
