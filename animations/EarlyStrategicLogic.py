from manim import *
import networkx as nx

class EarlyROIExplanation(Scene):
    """Explains the 'Next Best Step' logic from ROIAlg.py"""
    def construct(self):
        title = Text("ROI Strategy: The 'Next Best Step'", color=BLUE).scale(0.7)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        # Setup a linear path to show 'Next Step' logic
        node_pos = {0: [-4, 0, 0], 1: [-2, 0, 0], 2: [0, 0, 0], 3: [2, 0, 0], 4: [4, 0, 0]}
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        m_graph = Graph(list(node_pos.keys()), edges, layout=node_pos).shift(DOWN*0.5)
        self.play(Create(m_graph))

        # 1. STRATEGIC COST (Dijkstra)
        cost_text = Text("1. Calculate Strategic Cost (Dijkstra)", color=YELLOW).scale(0.5).next_to(title, DOWN)
        self.play(Write(cost_text))
        
        # Highlight path difficulty
        for u, v in edges[:3]:
            self.play(m_graph.edges[(u, v)].animate.set_color(YELLOW).set_stroke(width=8), run_time=0.4)
        
        cost_label = Text("Cumulative Difficulty", color=YELLOW).scale(0.4).next_to(m_graph.vertices[3], DOWN)
        self.play(Write(cost_label))
        self.wait(1)
        self.play(FadeOut(cost_text), FadeOut(cost_label))

        # 2. ROI CALCULATION
        roi_text = Text("2. Square the Points for ROI", color=GREEN).scale(0.5).next_to(title, DOWN)
        self.play(Write(roi_text))
        
        # Standard text formula to avoid LaTeX errors
        formula = Text("ROI = (Points^2) / Cost", color=GREEN).scale(0.6).to_edge(LEFT)
        self.play(Write(formula))
        
        # Target node gets big
        self.play(m_graph.vertices[4].animate.set_color(GREEN).scale(1.5))
        self.wait(2)
        self.play(FadeOut(roi_text), FadeOut(formula), FadeOut(m_graph))

class BARCSClusterExplanation(Scene):
    """Explains the Neighborhood Scouting logic from BARCS.py"""
    def construct(self):
        title = Text("BARCS: Cluster Scouting Strategy", color=BLUE).scale(0.7).to_edge(UP)
        self.play(Write(title))

        # Create a cluster-looking graph
        node_pos = {
            0: [-3, 0, 0], 
            1: [0, 0, 0], # Gateway Node
            2: [2, 1, 0], 3: [3, 0.5, 0], 4: [3, -0.5, 0], 5: [2, -1, 0] # The Cluster
        }
        edges = [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
        m_graph = Graph(list(node_pos.keys()), edges, layout=node_pos).shift(DOWN*0.5)
        self.play(Create(m_graph))

        # Identify Strategic Gateway
        gate_text = Text("Identifying the Gateway Node", color=ORANGE).scale(0.5).next_to(title, DOWN)
        self.play(Write(gate_text))
        self.play(m_graph.vertices[1].animate.set_color(ORANGE).scale(1.3))
        
        # Neighborhood Bonus visualization
        # Dashed circle to show scouting range
        base_circle = Circle(radius=1.8, color=BLUE).move_to(m_graph.vertices[1])
        scout_range = DashedVMobject(base_circle)
        self.play(Create(scout_range))
        
        scout_label = Text("Scouting Neighborhood Points", color=BLUE).scale(0.4).next_to(scout_range, UP)
        self.play(Write(scout_label))
        
        # Highlight strategic neighbors
        cluster_nodes = [m_graph.vertices[i] for i in range(2, 6)]
        self.play(*[Indicate(n, color=GOLD) for n in cluster_nodes])
        
        # Explain math simply
        math_label = Text("Total Utility = Base + (0.5 * Neighbors)", color=BLUE).scale(0.4).to_edge(LEFT)
        self.play(Write(math_label))
        self.wait(3)