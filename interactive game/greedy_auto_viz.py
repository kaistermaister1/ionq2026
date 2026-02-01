from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

import optimal


@dataclass(frozen=True)
class GreedyParams:
    # Edge base weights by difficulty bucket
    easy_edge_weight: float = 15.0      # D1/D2/D3
    hard_edge_weight: float = 100.0  # D4/D5 (discourage but still try)

    # Node value = points + bonuspoints
    # points := utility_qubits, bonuspoints := bonus_bell_pairs
    # Effective edge weight = base - node_value  (lower is better)
    max_consecutive_failures: int = 3
    # If false, the greedy will NEVER attempt D4/D5 edges (it will stop instead).
    allow_hard_edges: bool = True

    # UI/loop
    step_pause_seconds: float = 0.25  # small pause so UI updates


class GreedyAutoViz:
    """
    Auto-running greedy visualizer.

    Greedy policy (per your description):
    - Center node is the most recently captured one (on success)
    - Candidate edges are those from an owned node -> unowned-by-you node
    - Edge base weight:
        - D1/D2/D3 => 15
        - D4/D5 => 1_000_000
    - Effective weight = base_weight - (utility_qubits + bonus_bell_pairs of the target node)
      Choose the candidate edge with the MINIMUM effective weight.
    - If attacking a target node fails 3 times in a row, skip it and move on.

    Controls:
    - Space: start/stop auto-run
    - n: single-step
    - r: refit view
    - Scroll: zoom
    - Right-drag or Shift+left-drag: pan
    """

    def __init__(self, client, params: Optional[GreedyParams] = None):
        self.client = client
        self.params = params or GreedyParams()

        # Full graph (static) + metadata
        self.graph: nx.Graph = nx.Graph()
        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}

        # Dynamic status state
        self.status: Dict = {}
        self.owned_edges: Set[Tuple[str, str]] = set()
        # Reachability is based on owned_edges + starting_node connectivity
        self.reachable_nodes: Set[str] = set()
        # Claimable is a list of edge dicts from the SDK (connectivity-aware)
        self.claimable_edge_data: list[Dict] = []

        # Greedy state
        self.center_node: Optional[str] = None
        self.selected_edge: Optional[Tuple[str, str]] = None
        self.radius: int = 2
        self.fail_counts: Dict[str, int] = {}  # consecutive fails per target node
        self.skipped_targets: Set[str] = set()
        self.running: bool = False
        self.step_index: int = 0
        self.last_action: str = ""

        # Layout cache
        self.pos: Dict[str, Tuple[float, float]] = {}
        self._subgraph: nx.Graph = nx.Graph()
        self._layout_nodes: Optional[frozenset[str]] = None

        # Pan state (axis-only pan; no redraw)
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
        self._timer = None
        self._in_step: bool = False

        self._load_graph_once()
        self.refresh_status()

        starting = self.status.get("starting_node")
        if starting:
            self.center_node = starting
        elif self.reachable_nodes:
            self.center_node = next(iter(self.reachable_nodes))
        else:
            self.center_node = next(iter(self.graph.nodes), None)

    def _load_graph_once(self) -> None:
        graph_data = self.client.get_cached_graph()
        # If networking fails, GameClient returns {"_error": {...}}; bail early with a clear message.
        if isinstance(graph_data, dict) and graph_data.get("_error"):
            raise RuntimeError(f"Failed to fetch graph: {graph_data.get('_error')}")

        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()

        for node in graph_data.get("nodes", []):
            node_id = node["node_id"]
            self.nodes[node_id] = node
            self.graph.add_node(node_id)

        for edge in graph_data.get("edges", []):
            a, b = edge["edge_id"]
            self.edges[(a, b)] = edge
            self.edges[(b, a)] = edge
            self.graph.add_edge(a, b)

    def refresh_status(self) -> None:
        # IMPORTANT: For automation, use SERVER-TRUTH status.
        # Including optimistic edges can make us think we can expand from a node before the server
        # has processed it, which leads to EDGE_NOT_ADJACENT spam.
        self.status = self.client.get_status(include_optimistic=False) or {}

        self.owned_edges = set()
        for e in self.status.get("owned_edges", []):
            a, b = e
            self.owned_edges.add((a, b))
            self.owned_edges.add((b, a))

        self.reachable_nodes = set(self.client.get_reachable_nodes(self.status) or set())
        self.claimable_edge_data = list(self.client.get_claimable_edges() or [])

    def _visible_nodes(self) -> Set[str]:
        if not self.center_node or self.center_node not in self.graph:
            return set()
        dists = nx.single_source_shortest_path_length(self.graph, self.center_node, cutoff=self.radius)
        return set(dists.keys())

    def _fit_view_to_nodes(self) -> None:
        if not self.pos or not self._subgraph.nodes():
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

        nx0 = x - (x - x0) * scale
        nx1 = x + (x1 - x) * scale
        ny0 = y - (y - y0) * scale
        ny1 = y + (y1 - y) * scale

        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny0, ny1)

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 1 / 1.2 if event.button == "up" else 1.2
        self._zoom_at(event.xdata, event.ydata, scale)
        self.fig.canvas.draw_idle()

    def _start_pan(self, event) -> None:
        self._panning = True
        self._pan_moved = False
        self._pan_press = (float(event.xdata), float(event.ydata))
        self._pan_xlim = self.ax.get_xlim()
        self._pan_ylim = self.ax.get_ylim()

    def _stop_pan(self) -> None:
        self._panning = False
        self._pan_press = None
        self._pan_xlim = None
        self._pan_ylim = None

    def on_button_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        is_shift = bool(getattr(event, "key", None) == "shift")
        if event.button == 3 or (event.button == 1 and is_shift):
            self._start_pan(event)

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

        (xl0, xl1) = self._pan_xlim
        (yl0, yl1) = self._pan_ylim
        self.ax.set_xlim(xl0 - dx, xl1 - dx)
        self.ax.set_ylim(yl0 - dy, yl1 - dy)

        self._pan_moved = True
        self.fig.canvas.draw_idle()

    def on_button_release(self, event):
        if self._panning:
            moved = self._pan_moved
            self._stop_pan()
            if moved:
                return

    def _base_edge_weight(self, diff: int) -> float:
        if diff in (1, 2, 3):
            return self.params.easy_edge_weight
        return self.params.hard_edge_weight

    def _node_value(self, node_id: str) -> float:
        node = self.nodes.get(node_id, {})
        uq = float(node.get("utility_qubits", 0) or 0)
        bonus = float(node.get("bonus_bell_pairs", 0) or 0)
        return uq + bonus

    def _choose_greedy_target(self) -> Optional[Tuple[str, str]]:
        """
        Returns (u, v) where u is reachable-by-you and v is outside your reachable component.
        """
        candidates: list[Tuple[float, str, str]] = []
        for e in self.claimable_edge_data:
            try:
                a, b = e["edge_id"]
            except Exception:
                continue
            # Orient edge from reachable -> outside
            if (a in self.reachable_nodes) == (b in self.reachable_nodes):
                continue
            u, v = (a, b) if a in self.reachable_nodes else (b, a)

            if v in self.skipped_targets:
                continue
            if self.fail_counts.get(v, 0) >= self.params.max_consecutive_failures:
                continue

            diff = int(self.edges.get((u, v), {}).get("difficulty_rating", 999))
            if (diff >= 4) and (not self.params.allow_hard_edges):
                continue
            base = self._base_edge_weight(diff)
            node_value = self._node_value(v)
            eff = base - node_value
            candidates.append((eff, u, v))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        _, u, v = candidates[0]
        return (u, v)

    def _attack_edge(self, u: str, v: str) -> bool:
        edge_info = self.edges.get((u, v), {})
        diff = int(edge_info.get("difficulty_rating", 0))

        try:
            if diff == 1:
                result = optimal.run_d1_n2_on_edge(self.client, u, v)
            elif diff == 2:
                result = optimal.run_d2_on_edge(self.client, u, v)
            elif diff == 3:
                result = optimal.run_d3_n3_on_edge(self.client, u, v)
            elif diff == 4:
                result = optimal.conquer_edge_with_d4_n3(self.client, u, v)
            elif diff == 5:
                result = optimal.conquer_edge_with_d5_n5(self.client, u, v)
            else:
                print(f"No strategy for difficulty {diff} (skipping).")
                return False
        except Exception as e:
            print(f"Attack error on {u} <-> {v}: {e}")
            return False

        return bool(result and result.get("ok"))

    def step_once(self) -> None:
        self.refresh_status()
        self.step_index += 1

        target = self._choose_greedy_target()
        if target is None:
            hard_note = "" if self.params.allow_hard_edges else " (hard edges disabled)"
            self._set_bottom(f"[{self.step_index}] No greedy target available{hard_note}.")
            self.selected_edge = None
            self.redraw(preserve_view=True, keep_bottom=True)
            self.running = False
            if self._timer is not None:
                self._timer.stop()
            return

        u, v = target
        self.selected_edge = (u, v)
        diff = int(self.edges.get((u, v), {}).get("difficulty_rating", 999))
        base = self._base_edge_weight(diff)
        val = self._node_value(v)
        eff = base - val
        fails = self.fail_counts.get(v, 0)

        self._set_bottom(
            f"[{self.step_index}] target {u} -> {v} | D{diff} | eff={eff:g} | fails={fails}/{self.params.max_consecutive_failures}"
        )
        self.redraw(preserve_view=True, keep_bottom=True)
        plt.pause(self.params.step_pause_seconds)

        ok = self._attack_edge(u, v)
        if ok:
            print(f"Greedy: SUCCESS on {u} -> {v}")
            self.last_action = f"SUCCESS {u} -> {v} (D{diff})"
            self.fail_counts[v] = 0
            self.skipped_targets.discard(v)

            # Center on most recently captured node (per request)
            self.center_node = v

            # Layout changes because visible set changes; refit view
            self.refresh_status()
            self.selected_edge = None
            self.redraw(preserve_view=False)
        else:
            print(f"Greedy: FAIL on {u} -> {v}")
            self.last_action = f"FAIL {u} -> {v} (D{diff})"
            self.fail_counts[v] = self.fail_counts.get(v, 0) + 1
            if self.fail_counts[v] >= self.params.max_consecutive_failures:
                self.skipped_targets.add(v)
                print(f"Greedy: skipping {v} after {self.fail_counts[v]} consecutive failures")
                self.last_action += f" | SKIP after {self.fail_counts[v]}"

            # Keep same center/layout; just update styling/title
            self.refresh_status()
            self.redraw(preserve_view=True, keep_bottom=True)

    def _tick(self):
        if not self.running or self._in_step:
            return
        self._in_step = True
        try:
            self.step_once()
        finally:
            self._in_step = False

    def on_key(self, event):
        if event.key == " ":
            self.running = not self.running
            if self.running:
                self._set_bottom("Auto-run: ON (greedy)")
                self.redraw(preserve_view=True, keep_bottom=True)
                if self._timer is not None:
                    self._timer.start()
            else:
                self._set_bottom("Auto-run: OFF")
                self.redraw(preserve_view=True, keep_bottom=True)
                if self._timer is not None:
                    self._timer.stop()
            return

        if event.key == "n":
            self.step_once()
            return

        if event.key == "r":
            self.redraw(preserve_view=False)
            return

        if event.key in ["+", "="]:
            self.radius = min(6, self.radius + 1)
            self.redraw(preserve_view=False)
            return

        if event.key in ["-", "_"]:
            self.radius = max(1, self.radius - 1)
            self.redraw(preserve_view=False)
            return

    def _set_top(self, text: str) -> None:
        if self.fig is None:
            return
        if self._hud_top is None:
            self._hud_top = self.fig.text(
                0.5,
                0.99,
                "",
                ha="center",
                va="top",
                fontsize=9,
                family="monospace",
            )
        self._hud_top.set_text(text)

    def _set_bottom(self, text: str) -> None:
        if self.fig is None:
            return
        if self._hud_bottom is None:
            self._hud_bottom = self.fig.text(
                0.5,
                0.01,
                "",
                ha="center",
                va="bottom",
                fontsize=9,
                family="monospace",
            )
        self._hud_bottom.set_text(text)

    def _default_top(self) -> str:
        player = self.status.get("player_id", "?")
        name = self.status.get("name", "")
        score = self.status.get("score", 0)
        budget = self.status.get("budget", 0)
        center = self.center_node or "?"
        tail = f"Last: {self.last_action}" if self.last_action else ""
        return f"{player} ({name}) | Score: {score} | Budget: {budget} | Center: {center} | Radius: {self.radius} | {tail}"

    def _default_bottom(self) -> str:
        return "Space=run/pause | n=step | +/- radius | scroll=zoom | right-drag pan"

    def redraw(self, preserve_view: bool, keep_bottom: bool = False) -> None:
        xlim = ylim = None
        bottom_text = None
        if preserve_view and self.ax is not None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            if self._hud_bottom is not None:
                bottom_text = self._hud_bottom.get_text()

        self.ax.clear()
        self.render_graph()

        self._set_top(self._default_top())
        if keep_bottom and bottom_text:
            self._set_bottom(bottom_text)
        else:
            self._set_bottom(self._default_bottom())

        if preserve_view and xlim and ylim:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            self._fit_view_to_nodes()

        self.fig.canvas.draw_idle()

    def render_graph(self) -> None:
        visible = self._visible_nodes()
        if not visible:
            visible = set(self.graph.nodes())

        self._subgraph = self.graph.subgraph(visible).copy()

        nodes_key = frozenset(self._subgraph.nodes())
        need_layout = self._layout_nodes != nodes_key or not all(n in self.pos for n in nodes_key)
        if need_layout:
            initial_pos = {n: self.pos[n] for n in self._subgraph.nodes() if n in self.pos}
            self.pos = nx.spring_layout(
                self._subgraph,
                seed=42,
                pos=initial_pos if initial_pos else None,
                iterations=30,
            )
            self._layout_nodes = nodes_key

        # Edges
        edgelist = list(self._subgraph.edges())
        claimable_undirected: Set[frozenset[str]] = set()
        for e in self.claimable_edge_data:
            try:
                a, b = e["edge_id"]
            except Exception:
                continue
            claimable_undirected.add(frozenset((a, b)))

        edge_colors = []
        edge_widths = []

        # If we have a selected edge (u->v), highlight the reachable path to u as well.
        path_edges: Set[frozenset[str]] = set()
        if self.selected_edge:
            u, _v = self.selected_edge
            start = self.status.get("starting_node")
            if start and start in self._subgraph and u in self._subgraph:
                try:
                    owned_g = nx.Graph()
                    owned_g.add_node(start)
                    for (a, b) in self.owned_edges:
                        owned_g.add_edge(a, b)
                    path = nx.shortest_path(owned_g, start, u)
                    for i in range(len(path) - 1):
                        path_edges.add(frozenset((path[i], path[i + 1])))
                except Exception:
                    pass

        for (u, v) in edgelist:
            if self.selected_edge and set(self.selected_edge) == {u, v}:
                edge_colors.append("#E53935")
                edge_widths.append(4.0)
            elif frozenset((u, v)) in path_edges:
                edge_colors.append("#8E24AA")  # purple path-to-target
                edge_widths.append(3.2)
            elif frozenset((u, v)) in claimable_undirected:
                edge_colors.append("#FF9800")
                edge_widths.append(2.5)
            elif (u, v) in self.owned_edges:
                edge_colors.append("#4CAF50")
                edge_widths.append(2.0)
            else:
                edge_colors.append("#9E9E9E")
                edge_widths.append(1.0)

        nx.draw_networkx_edges(
            self._subgraph,
            self.pos,
            ax=self.ax,
            edgelist=edgelist,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.9,
        )

        # Nodes (same scheme as interactive_viz)
        nodelist = list(self._subgraph.nodes())
        node_colors = []
        for n in nodelist:
            # show reachable component as "green" (you can expand from it)
            if n in self.reachable_nodes:
                node_colors.append("#4CAF50")
                continue
            bonus = int(self.nodes.get(n, {}).get("bonus_bell_pairs", 0) or 0)
            if bonus > 0:
                node_colors.append("#FFEB3B")
            else:
                node_colors.append("#2196F3")

        node_sizes = [220 + self.nodes.get(n, {}).get("utility_qubits", 1) * 35 for n in nodelist]
        nx.draw_networkx_nodes(
            self._subgraph,
            self.pos,
            ax=self.ax,
            nodelist=nodelist,
            node_color=node_colors,
            node_size=node_sizes,
            linewidths=0.6,
            edgecolors="#111111",
        )

        # Node labels: name + points/bonus + fail count if any
        labels = {}
        for n in nodelist:
            node = self.nodes.get(n, {})
            uq = node.get("utility_qubits", "?")
            bonus = int(node.get("bonus_bell_pairs", 0) or 0)
            fails = self.fail_counts.get(n, 0)
            skipped = " (SKIP)" if n in self.skipped_targets else ""
            line2 = f"+{uq}"
            # "uncaptured by you" => outside your reachable component (you don't have a path yet)
            if n not in self.reachable_nodes and bonus > 0:
                line2 += f" (+{bonus}B)"
            line3 = f"fails:{fails}{skipped}" if (fails > 0 or skipped) else ""
            labels[n] = f"{n}\n{line2}" + (f"\n{line3}" if line3 else "")

        nx.draw_networkx_labels(
            self._subgraph,
            self.pos,
            ax=self.ax,
            labels=labels,
            font_size=7,
        )

        # Edge difficulty labels (D1..)
        for (u, v) in edgelist:
            diff = self.edges.get((u, v), {}).get("difficulty_rating", None)
            if diff is None:
                continue
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            self.ax.text(
                mx,
                my,
                f"D{diff}",
                fontsize=7,
                ha="center",
                va="center",
                color="#111111",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=5,
            )

        self.ax.axis("off")

    def show(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.render_graph()
        self._set_top(self._default_top())
        self._set_bottom(self._default_bottom())
        self._fit_view_to_nodes()

        # Stable repeating timer for auto-run (no recursive timers).
        self._timer = self.fig.canvas.new_timer(interval=1500)
        self._timer.add_callback(self._tick)

        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        plt.show()


if __name__ == "__main__":
    from client import GameClient
    import json
    from pathlib import Path

    SESSION_FILE = Path("session.json")

    def load_session():
        if not SESSION_FILE.exists():
            return None
        with open(SESSION_FILE) as f:
            data = json.load(f)
        c = GameClient(api_token=data.get("api_token"))
        c.player_id = data.get("player_id")
        c.name = data.get("name")
        return c

    client = load_session()
    if not client:
        print("Please run gameplay.py first to register/login!")
    else:
        print(f"Loaded session for {client.player_id}")
        GreedyAutoViz(client).show()

