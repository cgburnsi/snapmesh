"""
snapfvm/grid.py
---------------
The static, compiled representation of the mesh.
Optimized for solver speed (Arrays).
UPDATED: Calculates Normals, Areas, and Volumes.
"""
import numpy as np

class Grid:
    def __init__(self, mesh_obj):
        """
        Compiles a snapmesh.mesh.Mesh object into a solver-ready Grid.
        """
        print(f"--- Compiling Grid ---")
        
        # 1. Sort objects for deterministic indices
        # We sort by ID to ensure every run produces identical arrays
        sorted_nodes = sorted(mesh_obj.nodes.values(), key=lambda n: n.id)
        sorted_edges = sorted(mesh_obj.edges.values(), key=lambda e: e.id)
        sorted_cells = sorted(mesh_obj.cells.values(), key=lambda c: c.id)
        
        self.n_nodes = len(sorted_nodes)
        self.n_faces = len(sorted_edges)
        self.n_cells = len(sorted_cells)
        
        print(f"   -> Elements: {self.n_nodes} Nodes, {self.n_faces} Faces, {self.n_cells} Cells")

        # --- MAPS (Object ID -> Array Index) ---
        node_map = {n.id: i for i, n in enumerate(sorted_nodes)}
        edge_map = {e.id: i for i, e in enumerate(sorted_edges)}
        
        # --- 2. NODE ARRAYS ---
        self.nodes_x = np.zeros(self.n_nodes, dtype=float)
        self.nodes_y = np.zeros(self.n_nodes, dtype=float)
        for i, n in enumerate(sorted_nodes):
            self.nodes_x[i] = n.x
            self.nodes_y[i] = n.y

        # --- 3. CELL GEOMETRY (Centroids & Volumes) ---
        # We calculate this FIRST so we can orient normals correctly later
        self.cell_centers = np.zeros((self.n_cells, 2), dtype=float)
        self.cell_volumes = np.zeros(self.n_cells, dtype=float)
        
        for i, c in enumerate(sorted_cells):
            # 2D Triangle Volume = 0.5 * |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
            # We assume triangular cells for the Generator, but Grid handles polygons?
            # For robustness, we use the nodes provided by the cell.
            n1, n2, n3 = c.n1, c.n2, c.n3
            
            # Centroid
            cx = (n1.x + n2.x + n3.x) / 3.0
            cy = (n1.y + n2.y + n3.y) / 3.0
            self.cell_centers[i] = [cx, cy]
            
            # Area (Volume)
            area = 0.5 * np.abs((n2.x - n1.x)*(n3.y - n1.y) - (n3.x - n1.x)*(n2.y - n1.y))
            self.cell_volumes[i] = area

        # --- 4. FACE CONNECTIVITY & GEOMETRY ---
        self.face_nodes = np.zeros((self.n_faces, 2), dtype=int)
        self.face_cells = np.full((self.n_faces, 2), -1, dtype=int) # -1 = Boundary
        self.face_normals = np.zeros((self.n_faces, 2), dtype=float)
        self.face_midpoints = np.zeros((self.n_faces, 2), dtype=float)
        
        # Step A: Build raw connectivity from Edges
        for i, e in enumerate(sorted_edges):
            self.face_nodes[i, 0] = node_map[e.n1.id]
            self.face_nodes[i, 1] = node_map[e.n2.id]
            
            # Calculate Raw Geometry (assuming n1 -> n2 direction)
            x1, y1 = e.n1.x, e.n1.y
            x2, y2 = e.n2.x, e.n2.y
            
            mid_x = 0.5 * (x1 + x2)
            mid_y = 0.5 * (y1 + y2)
            self.face_midpoints[i] = [mid_x, mid_y]
            
            # Normal: (dy, -dx) gives a vector rotated -90 deg (to the right)
            dx, dy = x2 - x1, y2 - y1
            # Length is embedded in the normal magnitude
            self.face_normals[i] = [dy, -dx] 

        # Step B: Populate Face Neighbors (Iterate Cells)
        for c_idx, cell in enumerate(sorted_cells):
            for edge in [cell.e1, cell.e2, cell.e3]:
                if edge is None: continue
                f_idx = edge_map[edge.id]
                
                if self.face_cells[f_idx, 0] == -1:
                    self.face_cells[f_idx, 0] = c_idx
                else:
                    self.face_cells[f_idx, 1] = c_idx

        # Step C: Re-orient Normals (CRITICAL)
        # Normal must point from Left Cell (0) -> Right Cell (1)
        # If Right is -1, it must point OUT of Left.
        for i in range(self.n_faces):
            c_left = self.face_cells[i, 0]
            
            if c_left == -1:
                # Should not happen if logic is correct, but swap if 0 is empty
                self.face_cells[i, 0] = self.face_cells[i, 1]
                self.face_cells[i, 1] = -1
                c_left = self.face_cells[i, 0]

            # Vector from Left Cell Center to Face Midpoint
            # d = midpoint - cell_center
            cc = self.cell_centers[c_left]
            fm = self.face_midpoints[i]
            dx = fm[0] - cc[0]
            dy = fm[1] - cc[1]
            
            # Dot product with current normal
            dot = dx * self.face_normals[i, 0] + dy * self.face_normals[i, 1]
            
            if dot < 0:
                # Normal points INTO the left cell. Flip it.
                self.face_normals[i] *= -1.0
                
        print(f"   -> Geometry built. Normals oriented Left->Right.")