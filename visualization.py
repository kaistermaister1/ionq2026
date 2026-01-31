"""
Visualization - Client-side visualization utilities.

This module provides visualization of the game graph from JSON data
received from the server. It has NO game logic - purely for display.
"""

from typing import Dict, List, Optional, Set, Tuple
import networkx as nx

# Optional: matplotlib for rendering (not required for basic functionality)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GraphTool:
    """
    Client-side graph visualization tool.

    Constructed from JSON data received from server.get_graph_info().
    Provides methods for rendering and querying the public graph state.
    """

    def __init__(self, graph_data: Optional[Dict] = None):
        """
        Initialize the graph tool.

        Args:
            graph_data: Optional JSON dict from server.get_graph_info()
        """
        self.graph: nx.Graph = nx.Graph()
        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}

        if graph_data:
            self.load_from_json(graph_data)

    def load_from_json(self, graph_data: Dict) -> None:
        """
        Load graph from server JSON response.

        Args:
            graph_data: Dict with 'nodes' and 'edges' from server.get_graph_info()
        """
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()

        # Load nodes
        for node in graph_data.get("nodes", []):
            node_id = node["node_id"]
            self.nodes[node_id] = node
            self.graph.add_node(node_id)

        # Load edges
        for edge in graph_data.get("edges", []):
            edge_id = tuple(edge["edge_id"])
            self.edges[edge_id] = edge
            self.edges[(edge_id[1], edge_id[0])] = edge  # Reverse lookup
            self.graph.add_edge(edge_id[0], edge_id[1])

    # ==================== Query Methods ====================

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node data."""
        return self.nodes.get(node_id)

    def get_edge(self, node1: str, node2: str) -> Optional[Dict]:
        """Get edge data."""
        return self.edges.get((node1, node2))

    def get_all_nodes(self) -> List[str]:
        """Get all node IDs."""
        return list(self.nodes.keys())

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all unique edges."""
        seen = set()
        edges = []
        for edge_id in self.edges.keys():
            normalized = tuple(sorted(edge_id))
            if normalized not in seen:
                seen.add(normalized)
                edges.append(edge_id)
        return edges

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes."""
        if node_id not in self.graph:
            return []
        return list(self.graph.neighbors(node_id))

    def get_claimable_edges(self, subgraph_nodes: Set[str]) -> List[Tuple[str, str]]:
        """
        Get edges that can be claimed from a given subgraph.

        Args:
            subgraph_nodes: Set of node IDs in player's subgraph

        Returns:
            List of (node1, node2) tuples for edges adjacent to subgraph
        """
        claimable = []
        for node_id in subgraph_nodes:
            for neighbor in self.get_neighbors(node_id):
                if neighbor not in subgraph_nodes:
                    claimable.append((node_id, neighbor))
        return claimable

    def get_neighborhood(self, center_nodes: Set[str], radius: int = 1) -> Set[str]:
        """
        Get all nodes within N hops of the center nodes.

        Args:
            center_nodes: Set of node IDs to use as starting points
            radius: Number of hops to include (1 = immediate neighbors, 2 = neighbors of neighbors, etc.)

        Returns:
            Set of all node IDs within radius hops
        """
        if not center_nodes:
            return set()

        neighborhood = set(center_nodes)
        current_frontier = set(center_nodes)

        for _ in range(radius):
            next_frontier = set()
            for node_id in current_frontier:
                neighbors = self.get_neighbors(node_id)
                next_frontier.update(neighbors)
            neighborhood.update(next_frontier)
            current_frontier = next_frontier - neighborhood.union(center_nodes)

        return neighborhood

    # ==================== Visualization ====================

    def render(
        self,
        player_subgraph: Optional[Set[str]] = None,
        highlight_claimable: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        focus_radius: Optional[int] = None,
    ) -> None:
        """
        Render the graph using matplotlib.

        Args:
            player_subgraph: Set of node IDs owned by player (highlighted)
            highlight_claimable: Whether to highlight claimable edges
            figsize: Figure size tuple
            save_path: If provided, save figure to this path
            focus_radius: If provided, only show nodes within N hops of owned nodes.
                         If None and player_subgraph is provided, defaults to 2.
                         Set to -1 to show entire graph.
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        player_subgraph = player_subgraph or set()

        # Auto-enable focused view when player has nodes
        if focus_radius is None and player_subgraph:
            focus_radius = 2

        # Create focused subgraph if requested
        if focus_radius is not None and focus_radius >= 0 and player_subgraph:
            visible_nodes = self.get_neighborhood(player_subgraph, focus_radius)
            # Create a subgraph view
            graph_to_render = self.graph.subgraph(visible_nodes)
        else:
            graph_to_render = self.graph
            visible_nodes = set(self.graph.nodes())

        claimable_edges = (
            self.get_claimable_edges(player_subgraph) if highlight_claimable else []
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Layout
        pos = nx.spring_layout(graph_to_render, seed=42)

        # Node colors
        node_colors = []
        for node_id in graph_to_render.nodes():
            if node_id in player_subgraph:
                node_colors.append("#4CAF50")  # Green for owned
            else:
                node_colors.append("#2196F3")  # Blue for unowned

        # Node sizes based on utility qubits
        node_sizes = []
        for node_id in graph_to_render.nodes():
            node = self.nodes.get(node_id, {})
            utility = node.get("utility_qubits", 1)
            node_sizes.append(300 + utility * 100)

        # Draw nodes
        nx.draw_networkx_nodes(
            graph_to_render, pos, node_color=node_colors, node_size=node_sizes, ax=ax
        )

        # Edge colors
        edge_colors = []
        edge_widths = []
        for u, v in graph_to_render.edges():
            if (u, v) in claimable_edges or (v, u) in claimable_edges:
                edge_colors.append("#FF9800")  # Orange for claimable
                edge_widths.append(3)
            elif u in player_subgraph and v in player_subgraph:
                edge_colors.append("#4CAF50")  # Green for owned
                edge_widths.append(2)
            else:
                edge_colors.append("#9E9E9E")  # Gray for other
                edge_widths.append(1)

        # Draw edges
        nx.draw_networkx_edges(
            graph_to_render, pos, edge_color=edge_colors, width=edge_widths, ax=ax
        )

        # Node labels
        labels = {}
        for node_id in graph_to_render.nodes():
            node = self.nodes.get(node_id, {})
            utility = node.get("utility_qubits", "?")
            labels[node_id] = f"{node_id}\n({utility})"
        nx.draw_networkx_labels(graph_to_render, pos, labels, font_size=8, ax=ax)

        # Edge labels (difficulty rating)
        edge_labels = {}
        for u, v in graph_to_render.edges():
            edge = self.get_edge(u, v)
            if edge:
                difficulty = edge.get("difficulty_rating", "?")
                edge_labels[(u, v)] = f"D{difficulty}"
        nx.draw_networkx_edge_labels(
            graph_to_render, pos, edge_labels, font_size=7, ax=ax
        )

        # Title shows focus info
        if focus_radius is not None and focus_radius >= 0 and player_subgraph:
            title = f"Quantum Network Graph (Focused: {len(visible_nodes)}/{len(self.nodes)} nodes, radius={focus_radius})"
        else:
            title = "Quantum Network Graph (Full View)"
        ax.set_title(title)
        ax.axis("off")

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                      markersize=10, label='Owned nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                      markersize=10, label='Unowned nodes'),
            plt.Line2D([0], [0], color='#FF9800', linewidth=3, label='Claimable edges'),
            plt.Line2D([0], [0], color='#9E9E9E', linewidth=1, label='Other edges'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def render_focused(
        self,
        player_subgraph: Set[str],
        radius: int = 2,
        figsize: Tuple[int, int] = (10, 7),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Render a focused view around the player's subgraph.

        This is a convenience method that calls render() with focus_radius set.
        Only shows nodes within N hops of owned nodes.

        Args:
            player_subgraph: Set of node IDs owned by player
            radius: Number of hops to include (default: 2)
            figsize: Figure size tuple
            save_path: If provided, save figure to this path
        """
        self.render(
            player_subgraph=player_subgraph,
            highlight_claimable=True,
            figsize=figsize,
            save_path=save_path,
            focus_radius=radius,
        )

    def print_summary(self, player_subgraph: Optional[Set[str]] = None, focused: bool = True, radius: int = 2) -> None:
        """
        Print a text summary of the graph state.

        Args:
            player_subgraph: Optional set of owned node IDs
            focused: If True, only show nodes within radius hops of owned nodes
            radius: Number of hops to include when focused (default: 2)
        """
        player_subgraph = player_subgraph or set()

        # Determine which nodes to display
        if focused and player_subgraph:
            visible_nodes = self.get_neighborhood(player_subgraph, radius)
            nodes_to_show = {k: v for k, v in self.nodes.items() if k in visible_nodes}
        else:
            visible_nodes = set(self.nodes.keys())
            nodes_to_show = self.nodes

        print("=" * 50)
        if focused and player_subgraph:
            print(f"GRAPH SUMMARY (Focused: radius={radius})")
        else:
            print("GRAPH SUMMARY (Full View)")
        print("=" * 50)
        print(f"Total nodes: {len(self.nodes)} (showing: {len(nodes_to_show)})")
        print(f"Total edges: {len(self.get_all_edges())}")
        print(f"Owned nodes: {len(player_subgraph)}")
        print()

        print("NODES:")
        for node_id, node in sorted(nodes_to_show.items()):
            owned = "✓" if node_id in player_subgraph else " "
            print(f"  [{owned}] {node_id}: {node.get('utility_qubits', '?')} qubits, "
                  f"+{node.get('bonus_bell_pairs', 0)} bell pairs")
        print()

        if player_subgraph:
            claimable = self.get_claimable_edges(player_subgraph)
            print("CLAIMABLE EDGES:")
            if claimable:
                for u, v in claimable:
                    edge = self.get_edge(u, v)
                    if edge:
                        print(f"  {u} -- {v}: threshold={edge.get('base_threshold', '?'):.2f}, "
                              f"difficulty={edge.get('difficulty_rating', '?')}, "
                              f"capacity={edge.get('capacity', '?')}")
            else:
                print("  (none)")
        print("=" * 50)
