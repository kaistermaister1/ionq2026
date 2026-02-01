from manim import *
import numpy as np

class StrategicNetwork(MovingCameraScene):
    def construct(self):
        # ---------------------------------------------------------
        # 1. SETUP THE DATA
        # ---------------------------------------------------------
        vertices = [0]
        edges = []
        
        # Layer 1
        layer_1 = [1, 2, 3, 4, 5]
        vertices.extend(layer_1)
        for n in layer_1:
            edges.append((0, n))
            
        # Layer 2
        layer_2 = []
        current_id = 6
        for parent in layer_1:
            for _ in range(3):
                vertices.append(current_id)
                edges.append((parent, current_id))
                layer_2.append(current_id)
                current_id += 1
                
        # Layer 3 (The outer fringe)
        for parent in layer_2[:10]: 
            vertices.append(current_id)
            edges.append((parent, current_id))
            current_id += 1

        best_node = 8
        path_nodes = [0, 2, 8]
        path_edges = [(0, 2), (2, 8)]

        # ---------------------------------------------------------
        # 2. DRAW THE LARGE GRAPH
        # ---------------------------------------------------------
        initial_positions = {0: np.array([0, 0, 0])}
        
        layout_config = {
            "fixed": [0],             
            "pos": initial_positions,
            "dim": 3,
            "seed": 42 
        }
        
        g = Graph(
            vertices, 
            edges, 
            layout="spring", 
            layout_scale=8.0, # CRITICAL: This makes the tree huge
            layout_config=layout_config,
            vertex_config={0: {"color": GREEN, "radius": 0.2}},
            labels=False 
        )
        
        for v in vertices:
            if v != 0:
                g.vertices[v].set_color(GREY)
                g.vertices[v].scale(0.8)

        # Start zoomed out a bit to see the initial creation
        self.camera.frame.save_state()
        self.play(Create(g), run_time=3)
        self.wait()

        # ---------------------------------------------------------
        # 3. DYNAMIC BUDGET CUTOFF & CAMERA ZOOM
        # ---------------------------------------------------------
        # We define a radius that is small relative to the 8.0 scale
        cutoff_radius = 4.0 
        cutoff_circle = Circle(radius=cutoff_radius, color=RED, stroke_width=6)
        cutoff_circle.set_fill(RED, opacity=0.05)
        
        cutoff_label = Text("Dynamic Budget Cutoff", font_size=36, color=RED)
        cutoff_label.next_to(cutoff_circle, LEFT, buff=0.5)

        # Step 1: Show the circle
        self.play(Create(cutoff_circle), Write(cutoff_label))
        self.wait(0.5)

        # Step 2: Zoom the camera into the circle
        # We set the frame width to be just slightly larger than the circle
        self.play(
            self.camera.frame.animate.set_width(cutoff_circle.width * 1.3).move_to(cutoff_circle),
            run_time=2.5,
            rate_func=exponential_decay
        )
        
        # Calculate nodes to cull
        nodes_outside = []
        for v in vertices:
            pos = g.vertices[v].get_center()
            if np.linalg.norm(pos) > cutoff_radius:
                nodes_outside.append(v)

        # Fade the "Strategic Neighborhood" outsiders
        cull_anims = []
        for v in nodes_outside:
            cull_anims.append(g.vertices[v].animate.set_opacity(0.05))
            for e in edges:
                if v in e:
                    edge_obj = g.edges.get(e) or g.edges.get((e[1], e[0]))
                    if edge_obj:
                        cull_anims.append(edge_obj.animate.set_opacity(0.05))
        
        if cull_anims:
            self.play(*cull_anims, run_time=1.5)
        self.wait()

        # ---------------------------------------------------------
        # 4. STRATEGIC CENTERS
        # ---------------------------------------------------------
        strategic_nodes = [6, 10, best_node]
        s_anims = []
        for n in strategic_nodes:
            if n in g.vertices:
                s_anims.append(g.vertices[n].animate.set_color(BLUE).scale(2))
            
        if best_node in g.vertices:
            s_anims.append(g.vertices[best_node].animate.set_color(DARK_BLUE).scale(3.0))
        
        # Positioning the label relative to the moving camera frame
        centers_label = Text("Strategic Centers Identified", font_size=30, color=BLUE)
        centers_label.move_to(self.camera.frame.get_corner(UL) + RIGHT*2.5 + DOWN*0.8)

        self.play(*s_anims, Write(centers_label))
        self.wait()

        # ---------------------------------------------------------
        # 5. PATH & OPTIMAL STEP
        # ---------------------------------------------------------
        path_anims = []
        for edge in path_edges:
            edge_obj = g.edges.get(edge) or g.edges.get((edge[1], edge[0]))
            if edge_obj:
                path_anims.append(edge_obj.animate.set_color(YELLOW).set_stroke(width=10))
        
        for node in path_nodes:
            if node != 0:
                path_anims.append(g.vertices[node].animate.set_color(YELLOW))

        self.play(*path_anims, run_time=1.5)
        
        # Optimal Step focus
        first_step_node = path_nodes[1] 
        selection_circle = Circle(radius=0.6, color=GREEN, stroke_width=8).move_to(g.vertices[first_step_node])
        step_label = Text("Optimal Step", font_size=28, color=GREEN).next_to(selection_circle, DOWN)
        
        self.play(
            Create(selection_circle),
            g.vertices[first_step_node].animate.set_color(GREEN),
            Write(step_label)
        )
        
        self.wait(3)