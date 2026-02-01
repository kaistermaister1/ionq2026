from manim import *
import networkx as nx

class QuantumStrategyAnalysis(Scene):
    def construct(self):
        # --- TITLE ---
        title = Text("Overview of the BARCS Algorithm", color=BLUE).scale(0.8)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # --- 1. THE BASIC ROI MODEL ---
        roi_text = Text("1. ROI Model: Point Squaring", color=YELLOW).scale(0.6).next_to(title, DOWN)
        self.play(Write(roi_text))

        # Define a 'Cluster' Graph: Starting Hub (0-4) and a Bridge (5)
        # We manually set positions to ensure they look like clusters
        node_pos = {
            0: [-3, 1, 0], 1: [-4, 0.5, 0], 2: [-3.5, -0.5, 0], 3: [-2.5, -0.5, 0], 4: [-2, 0.5, 0], # Cluster A
            5: [0, 0, 0] # The Bridge Node
        }
        edges = [(0,1), (0,2), (0,3), (0,4), (4,5)]
        
        m_graph = Graph(list(node_pos.keys()), edges, layout=node_pos).shift(DOWN*0.5)
        self.play(Create(m_graph))

        eq_roi = Text("ROI = (Points^2) / Cost", color=YELLOW).scale(0.5).to_edge(LEFT).shift(UP*1)
        self.play(Write(eq_roi))
        
        # Highlight high-value target in local hub
        node_local_best = m_graph.vertices[1]
        label_local = Text("10 pts", color=GREEN).scale(0.4).next_to(node_local_best, UP)
        
        self.play(Write(label_local))
        self.play(node_local_best.animate.set_color(GREEN).scale(1.3))
        self.wait(1.5)
        
        self.play(FadeOut(roi_text), FadeOut(eq_roi), FadeOut(label_local))

        # --- 2. BARCS: CLUSTER SCOUTING ---
        barcs_text = Text("2. BARCS: Cluster Scouting", color=BLUE).scale(0.6).next_to(title, DOWN)
        self.play(Write(barcs_text))
        
        # Add the 'Goldmine' Cluster (6-11) connected to the bridge
        goldmine_pos = {
            6: [2, 0, 0], # Gateway to Goldmine
            7: [3, 1, 0], 8: [4, 0.5, 0], 9: [4, -0.5, 0], 10: [3, -1, 0], 11: [3.5, 0, 0]
        }
        goldmine_edges = [(5,6), (6,7), (6,8), (6,9), (6,10), (6,11), (7,8), (8,9), (9,10)]
        
        # Add Cluster B
        self.play(m_graph.animate.add_vertices(*goldmine_pos.keys(), positions=goldmine_pos))
        self.play(m_graph.animate.add_edges(*goldmine_edges))
        
        # Demonstrate Scouting logic: Gateway Node 6
        base_circle = Circle(radius=1.2, color=BLUE).move_to(m_graph.vertices[6])
        cluster_circle = DashedVMobject(base_circle)
        scout_label = Text("Scouting High-Value Cluster", color=BLUE).scale(0.4).next_to(cluster_circle, UP)
        
        self.play(Create(cluster_circle), Write(scout_label))
        # ROI brain sees total points behind the bridge
        gold_nodes = [m_graph.vertices[i] for i in range(7, 12)]
        self.play(*[Indicate(n, color=GOLD) for n in gold_nodes])
        self.wait(2)

        self.play(FadeOut(barcs_text), FadeOut(cluster_circle), FadeOut(scout_label))

        # --- 3. GREEDY BARCS: THE HYBRID ---
        hybrid_text = Text("3. Greedy BARCS: Mode Switching", color=ORANGE).scale(0.6).next_to(title, DOWN)
        self.play(Write(hybrid_text))

        mode_box = Rectangle(height=0.8, width=3, color=WHITE).to_edge(RIGHT).shift(UP*1)
        mode_label = Text("MODE: SCOUTING", color=WHITE).scale(0.4).move_to(mode_box)
        self.play(Create(mode_box), Write(mode_label))
        
        # Cross the bridge
        path = [(0,4), (4,5), (5,6)]
        for u, v in path:
            self.play(m_graph.edges[(u, v)].animate.set_color(ORANGE).set_stroke(width=8), run_time=0.4)
        
        # SWITCH TO GREEDY
        new_mode_label = Text("MODE: GREEDY", color=ORANGE).scale(0.4).move_to(mode_box)
        self.play(Transform(mode_label, new_mode_label), mode_box.animate.set_color(ORANGE))
        
        # Clean out the goldmine
        for target in range(7, 12):
            self.play(m_graph.edges[(6, target)].animate.set_color(ORANGE), 
                      m_graph.vertices[target].animate.set_color(ORANGE), run_time=0.25)

        self.wait(2)

        # --- 4. DESPERATION DASH ---
        dash_text = Text("4. Desperation Dash", color=RED).scale(0.6).next_to(title, DOWN)
        self.play(FadeOut(hybrid_text), Write(dash_text))
        
        budget_label = Text("Budget: 2 Pairs", color=RED).scale(0.5).next_to(mode_box, DOWN)
        self.play(Write(budget_label))
        
        final_mode_label = Text("MODE: DASH", color=RED).scale(0.4).move_to(mode_box)
        self.play(Transform(mode_label, final_mode_label), mode_box.animate.set_color(RED))
        
        # Ignore ROI, just grab closest cheap node (Node 2)
        self.play(Indicate(m_graph.vertices[2], color=RED))
        self.play(m_graph.edges[(0, 2)].animate.set_color(RED), m_graph.vertices[2].animate.set_color(RED))

        self.wait(3)
        self.play(FadeOut(m_graph), FadeOut(title), FadeOut(dash_text), 
                  FadeOut(mode_box), FadeOut(mode_label), FadeOut(budget_label))