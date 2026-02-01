from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx

import optimal


class InteractiveGraphTool:
    """
    Simple + robust interactive map.

    Behavior intentionally matches the stable bot logic in `BARCS.py`:
    - "Captured" nodes are whatever the server returns in `owned_nodes` (green).
    - Attackable edges are whatever `client.get_claimable_edges()` returns.
      (This prevents EDGE_NOT_ADJACENT confusion.)
    - Re-attacking the same edge is allowed (reinforcement can be required).

    Controls:
    - Left click node: recenter
    - Left click orange edge: select
    - Enter: attack selected edge
    - Esc: clear selection
    - Scroll: zoom
    - Right-drag (or Shift+left-drag): pan
    - +/-: radius
    """

    def __init__(self, client, *, session_file: Optional["Path"] = None, starting_candidates: Optional[List[Dict]] = None):
        self.client = client
        self.session_file = session_file
        self.starting_candidates = starting_candidates or []

        # Static graph
        self.graph: nx.Graph = nx.Graph()
        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}

        # Stable layout (full graph), computed once
        self.pos: Dict[str, Tuple[float, float]] = {}

        # Dynamic status (server-truth)
        self.status: Dict = {}
        self.owned_nodes: Set[str] = set()
        self.owned_edges: Set[Tuple[str, str]] = set()      # undirected canonical
        self.claimable_edges: Set[Tuple[str, str]] = set()  # undirected canonical

        # View
        self.center_node: Optional[str] = None
        self.selected_edge: Optional[Tuple[str, str]] = None
        self.radius: int = 2
        self._subgraph: nx.Graph = nx.Graph()

        # Pan state
        self._panning: bool = False
        self._pan_press: Optional[Tuple[float, float]] = None
        self._pan_xlim: Optional[Tuple[float, float]] = None
        self._pan_ylim: Optional[Tuple[float, float]] = None
        self._pan_moved: bool = False

        # Matplotlib state
        self.fig = None
        self.ax = None
        self._hud_top = None
        self._hud_bottom = None
        self._btn_reset = None
        self._btn_refresh = None

        self._load_graph_once()
        self._compute_layout_once()
        self.refresh()

        start = self.status.get("starting_node")
        if start:
            self.center_node = start
        elif self.owned_nodes:
            self.center_node = next(iter(self.owned_nodes))
        else:
            self.center_node = next(iter(self.graph.nodes), None)

    @staticmethod
    def _norm_edge(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _load_graph_once(self) -> None:
        graph_data = self.client.get_cached_graph()
        if isinstance(graph_data, dict) and graph_data.get("_error"):
            raise RuntimeError(f"Failed to fetch graph: {graph_data.get('_error')}")

        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()

        for node in graph_data.get("nodes", []):
            nid = node["node_id"]
            self.nodes[nid] = node
            self.graph.add_node(nid)

        for edge in graph_data.get("edges", []):
            a, b = edge["edge_id"]
            self.edges[(a, b)] = edge
            self.edges[(b, a)] = edge
            self.graph.add_edge(a, b)

    def _compute_layout_once(self) -> None:
        self.pos = nx.spring_layout(self.graph, seed=42)

    def refresh(self) -> None:
        # Server-truth state (robust; avoids optimistic glitches)
        self.status = self.client.get_status(include_optimistic=False) or {}
        self.owned_nodes = set(self.status.get("owned_nodes", []) or [])

        self.owned_edges = set()
        for e in self.status.get("owned_edges", []) or []:
            try:
                a, b = e
            except Exception:
                continue
            self.owned_edges.add(self._norm_edge(a, b))

        claimable = set()
        for e in self.client.get_claimable_edges(include_optimistic=False) or []:
            try:
                a, b = e["edge_id"]
            except Exception:
                continue
            claimable.add(self._norm_edge(a, b))
        self.claimable_edges = claimable

    def _set_top(self, text: str) -> None:
        if self.fig is None:
            return
        if self._hud_top is None:
            self._hud_top = self.fig.text(0.5, 0.99, "", ha="center", va="top", fontsize=9, family="monospace")
        self._hud_top.set_text(text)

    def _set_bottom(self, text: str) -> None:
        if self.fig is None:
            return
        if self._hud_bottom is None:
            self._hud_bottom = self.fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9, family="monospace")
        self._hud_bottom.set_text(text)

    def _default_huds(self) -> Tuple[str, str]:
        pid = self.status.get("player_id", "?")
        name = self.status.get("name", "")
        score = self.status.get("score", 0)
        budget = self.status.get("budget", 0)
        top = f"{pid} ({name}) | Score: {score} | Budget: {budget} | Center: {self.center_node} | Radius: {self.radius}"
        bottom = "node click=recenter | orange edge click=select | Enter=attack | Esc=clear | scroll=zoom | right-drag pan | +/- radius"
        return top, bottom

    def _visible_nodes(self) -> Set[str]:
        if not self.center_node or self.center_node not in self.graph:
            return set(self.graph.nodes())
        dists = nx.single_source_shortest_path_length(self.graph, self.center_node, cutoff=self.radius)
        return set(dists.keys())

    def _fit_view_to_nodes(self) -> None:
        if self.ax is None or not self._subgraph.nodes():
            return
        xs = [self.pos[n][0] for n in self._subgraph.nodes() if n in self.pos]
        ys = [self.pos[n][1] for n in self._subgraph.nodes() if n in self.pos]
        if not xs or not ys:
            return
        pad = 0.15
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        dx = (x1 - x0) or 1.0
        dy = (y1 - y0) or 1.0
        self.ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
        self.ax.set_ylim(y0 - pad * dy, y1 + pad * dy)

    def _zoom_at(self, x: float, y: float, scale: float) -> None:
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        self.ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        self._zoom_at(event.xdata, event.ydata, scale)
        self.fig.canvas.draw_idle()

    def on_button_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        is_shift = bool(getattr(event, "key", None) == "shift")
        if event.button == 3 or (event.button == 1 and is_shift):
            self._panning = True
            self._pan_moved = False
            self._pan_press = (float(event.xdata), float(event.ydata))
            self._pan_xlim = self.ax.get_xlim()
            self._pan_ylim = self.ax.get_ylim()

    def on_motion(self, event):
        if not self._panning:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if not self._pan_press or not self._pan_xlim or not self._pan_ylim:
            return
        x0, y0 = self._pan_press
        dx = float(event.xdata) - x0
        dy = float(event.ydata) - y0
        xl0, xl1 = self._pan_xlim
        yl0, yl1 = self._pan_ylim
        self.ax.set_xlim(xl0 - dx, xl1 - dx)
        self.ax.set_ylim(yl0 - dy, yl1 - dy)
        self._pan_moved = True
        self.fig.canvas.draw_idle()

    def on_button_release(self, event):
        was_panning = self._panning
        moved = self._pan_moved
        self._panning = False
        self._pan_press = None
        self._pan_xlim = None
        self._pan_ylim = None
        self._pan_moved = False

        if was_panning and moved:
            return
        if event.button == 1:
            self.on_click(event)

    @staticmethod
    def _dist2(px: float, py: float, qx: float, qy: float) -> float:
        dx = px - qx
        dy = py - qy
        return dx * dx + dy * dy

    @staticmethod
    def _point_segment_dist2(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab_len2 = abx * abx + aby * aby
        if ab_len2 <= 1e-12:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = (apx * abx + apy * aby) / ab_len2
        t = max(0.0, min(1.0, t))
        cx = ax + t * abx
        cy = ay + t * aby
        return (px - cx) ** 2 + (py - cy) ** 2

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.refresh()

        px, py = float(event.xdata), float(event.ydata)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        view = max(abs(x1 - x0), abs(y1 - y0)) or 1.0
        node_r2 = (0.035 * view) ** 2
        edge_r2 = (0.02 * view) ** 2

        # Node recenter
        best_node = None
        best_d2 = float("inf")
        for n in self._subgraph.nodes():
            x, y = self.pos[n]
            d2 = self._dist2(px, py, x, y)
            if d2 < best_d2:
                best_d2 = d2
                best_node = n
        if best_node is not None and best_d2 <= node_r2:
            self.center_node = best_node
            self.selected_edge = None
            self.redraw(refit=True)
            return

        # Claimable edge select
        best_edge = None
        best_ed2 = float("inf")
        for u, v in self._subgraph.edges():
            if self._norm_edge(u, v) not in self.claimable_edges:
                continue
            ax, ay = self.pos[u]
            bx, by = self.pos[v]
            d2 = self._point_segment_dist2(px, py, ax, ay, bx, by)
            if d2 < best_ed2:
                best_ed2 = d2
                best_edge = (u, v)
        if best_edge is None or best_ed2 > edge_r2:
            return

        self.selected_edge = best_edge
        u, v = best_edge
        e = self.edges.get((u, v), {})
        self._set_bottom(f"Selected: {u} <-> {v} | D{e.get('difficulty_rating','?')} thr={e.get('base_threshold','?')}")
        self.redraw(refit=False, keep_bottom=True)

    def on_key(self, event):
        if event.key == "escape":
            self.selected_edge = None
            self.redraw(refit=False)
            return
        if event.key == "enter" and self.selected_edge:
            self.attack_selected_edge()
            return
        if event.key in ["+", "="]:
            self.radius = min(6, self.radius + 1)
            self.redraw(refit=True)
            return
        if event.key in ["-", "_"]:
            self.radius = max(1, self.radius - 1)
            self.redraw(refit=True)
            return

    def _wait_for_status_change(self, timeout_s: float = 4.0) -> None:
        # Poll briefly so UI updates deterministically after a successful claim.
        start = time.time()
        prev_nodes = set(self.status.get("owned_nodes", []) or [])
        prev_budget = self.status.get("budget", None)
        while time.time() - start < timeout_s:
            s = self.client.get_status(include_optimistic=False) or {}
            now_nodes = set(s.get("owned_nodes", []) or [])
            now_budget = s.get("budget", None)
            if now_nodes != prev_nodes or now_budget != prev_budget:
                self.status = s
                return
            time.sleep(0.4)

    def attack_selected_edge(self):
        u, v = self.selected_edge
        diff = int(self.edges.get((u, v), {}).get("difficulty_rating", 0))
        self._set_bottom(f"Attacking {u} <-> {v} (D{diff}) ...")
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

        try:
            if diff == 1:
                res = optimal.run_d1_n2_on_edge(self.client, u, v)
            elif diff == 2:
                res = optimal.run_d2_on_edge(self.client, u, v)
            elif diff == 3:
                res = optimal.run_d3_n3_on_edge(self.client, u, v)
            elif diff == 4:
                res = optimal.conquer_edge_with_d4_n3(self.client, u, v)
            elif diff == 5:
                res = optimal.conquer_edge_with_d5_n5(self.client, u, v)
            else:
                self._set_bottom(f"No strategy for difficulty {diff}")
                return
        except Exception as e:
            self._set_bottom(f"Error: {e}")
            return

        ok = bool(res and res.get("ok") and res.get("data", {}).get("success"))
        if ok:
            self._set_bottom("Attack SUCCESS. (Node turns green only when it appears in owned_nodes.)")
            self._wait_for_status_change()
        else:
            code = (res or {}).get("error", {}).get("code")
            msg = (res or {}).get("error", {}).get("message")
            self._set_bottom(f"Attack FAILED: {code} {msg}")

        self.refresh()
        self.selected_edge = None
        self.redraw(refit=False, keep_bottom=True)

    def _save_session(self) -> None:
        if not self.session_file:
            return
        try:
            import json

            payload = {
                "api_token": getattr(self.client, "api_token", None),
                "player_id": getattr(self.client, "player_id", None),
                "name": getattr(self.client, "name", None),
                "starting_candidates": self.starting_candidates,
            }
            self.session_file.write_text(json.dumps(payload))
        except Exception:
            # never crash UI on session saving
            pass

    def _prompt_select_starting_node(self) -> None:
        """
        Console prompt to select starting node.
        Uses cached starting_candidates if present, otherwise asks for node_id directly.
        """
        status = self.client.get_status(include_optimistic=False) or {}
        if status.get("starting_node"):
            return

        candidates = self.starting_candidates or []
        if candidates:
            print("\nStarting candidates:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c.get('node_id')} | +{c.get('utility_qubits', 0)} pts | +{c.get('bonus_bell_pairs', 0)} budget")
            raw = input("Pick starting node by index or exact node_id: ").strip()
            node_id = None
            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(candidates):
                    node_id = candidates[idx].get("node_id")
            if not node_id:
                node_id = raw
        else:
            node_id = input("Enter starting node_id (from your candidate list): ").strip()

        if not node_id:
            print("No starting node selected.")
            return

        res = self.client.select_starting_node(node_id)
        if not res.get("ok"):
            print("Failed to select starting node:", res.get("error", {}).get("code"), res.get("error", {}).get("message"))
            return

        # update view
        self.refresh()
        self.center_node = (self.client.get_status(include_optimistic=False) or {}).get("starting_node") or node_id

    def reset_account(self) -> None:
        """
        Reset game progress and re-select starting node.
        """
        confirm = input("Type YES to restart your account (resets score/budget and starting node): ").strip()
        if confirm != "YES":
            self._set_bottom("Reset cancelled.")
            self.fig.canvas.draw_idle()
            return

        self._set_bottom("Resetting... (check console)")
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

        res = self.client.restart()
        if not res.get("ok"):
            self._set_bottom(f"Reset FAILED: {(res.get('error') or {}).get('code')} {(res.get('error') or {}).get('message')}")
            self.fig.canvas.draw_idle()
            return

        # restart sometimes returns candidates; persist if present
        try:
            data = res.get("data", {}) or {}
            cands = data.get("starting_candidates")
            if isinstance(cands, list) and cands:
                self.starting_candidates = cands
                self._save_session()
        except Exception:
            pass

        # prompt re-selection (console)
        self._prompt_select_starting_node()
        self._save_session()

        self.refresh()
        if self.status.get("starting_node"):
            self.center_node = self.status.get("starting_node")
        self.redraw(refit=True)

    def refresh_and_redraw(self) -> None:
        self.refresh()
        self.redraw(refit=False)

    def redraw(self, refit: bool, keep_bottom: bool = False) -> None:
        xlim = ylim = None
        bottom = None
        if self.ax is not None and not refit:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            if keep_bottom and self._hud_bottom is not None:
                bottom = self._hud_bottom.get_text()

        self.ax.clear()
        self.render_graph()

        top, default_bottom = self._default_huds()
        self._set_top(top)
        self._set_bottom(bottom if (keep_bottom and bottom) else default_bottom)

        if xlim and ylim:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        elif refit:
            self._fit_view_to_nodes()
        self.fig.canvas.draw_idle()

    def render_graph(self) -> None:
        visible = self._visible_nodes()
        self._subgraph = self.graph.subgraph(visible).copy()

        # Edges
        edgelist = list(self._subgraph.edges())
        edge_colors = []
        edge_widths = []
        for (u, v) in edgelist:
            e = self._norm_edge(u, v)
            if self.selected_edge and set(self.selected_edge) == {u, v}:
                edge_colors.append("#E53935")
                edge_widths.append(4.0)
            elif e in self.claimable_edges:
                edge_colors.append("#FF9800")
                edge_widths.append(2.5)
            elif e in self.owned_edges:
                edge_colors.append("#4CAF50")
                edge_widths.append(2.0)
            else:
                edge_colors.append("#9E9E9E")
                edge_widths.append(1.0)

        nx.draw_networkx_edges(self._subgraph, self.pos, ax=self.ax, edgelist=edgelist, edge_color=edge_colors, width=edge_widths, alpha=0.9)

        # Nodes
        nodelist = list(self._subgraph.nodes())
        node_colors = []
        for n in nodelist:
            if n in self.owned_nodes:
                node_colors.append("#4CAF50")
            else:
                bonus = int(self.nodes.get(n, {}).get("bonus_bell_pairs", 0) or 0)
                node_colors.append("#FFEB3B" if bonus > 0 else "#2196F3")

        node_sizes = [220 + self.nodes.get(n, {}).get("utility_qubits", 1) * 35 for n in nodelist]
        nx.draw_networkx_nodes(self._subgraph, self.pos, ax=self.ax, nodelist=nodelist, node_color=node_colors, node_size=node_sizes, linewidths=0.6, edgecolors="#111111")

        # Labels
        labels = {}
        for n in nodelist:
            node = self.nodes.get(n, {})
            uq = node.get("utility_qubits", "?")
            bonus = int(node.get("bonus_bell_pairs", 0) or 0)
            line2 = f"+{uq}" + (f" (+{bonus}B)" if (n not in self.owned_nodes and bonus > 0) else "")
            labels[n] = f"{n}\n{line2}"
        nx.draw_networkx_labels(self._subgraph, self.pos, ax=self.ax, labels=labels, font_size=7)

        # Difficulty labels
        for (u, v) in edgelist:
            diff = self.edges.get((u, v), {}).get("difficulty_rating", None)
            if diff is None:
                continue
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            self.ax.text(mx, my, f"D{diff}", fontsize=7, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        self.ax.axis("off")

    def show(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.render_graph()
        top, bottom = self._default_huds()
        self._set_top(top)
        self._set_bottom(bottom)
        self._fit_view_to_nodes()

        # Buttons (top-left)
        ax_reset = self.fig.add_axes([0.01, 0.93, 0.10, 0.05])
        self._btn_reset = Button(ax_reset, "Reset")
        self._btn_reset.on_clicked(lambda _evt: self.reset_account())

        ax_refresh = self.fig.add_axes([0.12, 0.93, 0.10, 0.05])
        self._btn_refresh = Button(ax_refresh, "Refresh")
        self._btn_refresh.on_clicked(lambda _evt: self.refresh_and_redraw())

        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        plt.show()


if __name__ == "__main__":
    import json
    from pathlib import Path
    from client import GameClient

    SESSION_FILE = Path("session.json")

    def load_session():
        if not SESSION_FILE.exists():
            return None, []
        data = json.loads(SESSION_FILE.read_text())
        c = GameClient(api_token=data.get("api_token"))
        c.player_id = data.get("player_id")
        c.name = data.get("name")
        candidates = data.get("starting_candidates") or []
        return c, candidates

    def save_session(client: GameClient, candidates: List[Dict]):
        if not getattr(client, "api_token", None):
            return
        SESSION_FILE.write_text(
            json.dumps(
                {
                    "api_token": client.api_token,
                    "player_id": client.player_id,
                    "name": client.name,
                    "starting_candidates": candidates,
                }
            )
        )

    client, candidates = load_session()

    # If no saved session, register interactively.
    if not client or not getattr(client, "api_token", None):
        print("No saved session. Register a new player.")
        player_id = input("Player ID: ").strip()
        name = input("Name: ").strip()
        location = input("Location (remote or in_person): ").strip() or "remote"
        client = GameClient()
        reg = client.register(player_id, name, location=location)
        if not reg.get("ok"):
            raise RuntimeError(f"Register failed: {reg.get('error', {}).get('code')} {reg.get('error', {}).get('message')}")
        candidates = (reg.get("data", {}) or {}).get("starting_candidates", []) or []
        save_session(client, candidates)
        print("Registered and saved session.")

    # Ensure starting node selected
    status = client.get_status(include_optimistic=False) or {}
    if not status.get("starting_node"):
        if candidates:
            print("\nStarting candidates:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c.get('node_id')} | +{c.get('utility_qubits', 0)} pts | +{c.get('bonus_bell_pairs', 0)} budget")
            raw = input("Pick starting node by index or exact node_id: ").strip()
            node_id = None
            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(candidates):
                    node_id = candidates[idx].get("node_id")
            if not node_id:
                node_id = raw
        else:
            node_id = input("Enter starting node_id (from your candidate list): ").strip()

        sel = client.select_starting_node(node_id)
        if not sel.get("ok"):
            raise RuntimeError(f"Select starting node failed: {sel.get('error', {}).get('code')} {sel.get('error', {}).get('message')}")

    print(f"Loaded session for {client.player_id}")
    InteractiveGraphTool(client, session_file=SESSION_FILE, starting_candidates=candidates).show()

