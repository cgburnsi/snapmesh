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



































'''  The following commented-out code is the old version of the Grid class work.

import numpy as np
import math

class UnstructuredGrid:
    def __init__(self, mesh):
        self.num_cells = len(mesh.cells)
        self.num_nodes = len(mesh.nodes)
        
        # --- 1. Flatten Node Data ---
        self.nodes_x = np.zeros(self.num_nodes)
        self.nodes_y = np.zeros(self.num_nodes)
        self.node_id_map = {} 
        for i, (nid, n) in enumerate(mesh.nodes.items()):
            self.nodes_x[i] = n.x
            self.nodes_y[i] = n.y
            self.node_id_map[nid] = i

        # --- 2. Flatten Cell Data ---
        self.cell_id_map = {}
        self.cell_nodes = np.zeros((self.num_cells, 3), dtype=int)
        self.cell_centers_x = np.zeros(self.num_cells)
        self.cell_centers_y = np.zeros(self.num_cells)

        for i, (cid, cell) in enumerate(mesh.cells.items()):
            self.cell_id_map[cid] = i
            n1_idx = self.node_id_map[cell.n1.id]
            n2_idx = self.node_id_map[cell.n2.id]
            n3_idx = self.node_id_map[cell.n3.id]
            
            self.cell_nodes[i, 0] = n1_idx
            self.cell_nodes[i, 1] = n2_idx
            self.cell_nodes[i, 2] = n3_idx
            
            self.cell_centers_x[i] = (self.nodes_x[n1_idx] + self.nodes_x[n2_idx] + self.nodes_x[n3_idx]) / 3.0
            self.cell_centers_y[i] = (self.nodes_y[n1_idx] + self.nodes_y[n2_idx] + self.nodes_y[n3_idx]) / 3.0

        # --- 3. Build Connectivity (Faces) ---
        self.faces = []
        # NEW: We will store the boundary tag (string) for every face
        self.face_tags = [] 
        self._build_topology(mesh)
        
        # --- 4. Calculate Metrics ---
        self._calculate_geometric_properties()

    def _build_topology(self, mesh):
        edge_to_cells = {}
        for cid, cell in mesh.cells.items():
            c_idx = self.cell_id_map[cid]
            es = [tuple(sorted((cell.n1.id, cell.n2.id))),
                  tuple(sorted((cell.n2.id, cell.n3.id))),
                  tuple(sorted((cell.n3.id, cell.n1.id)))]
            for e in es:
                if e not in edge_to_cells: edge_to_cells[e] = []
                edge_to_cells[e].append(c_idx)

        num_raw_faces = len(edge_to_cells)
        self.face_nodes = np.zeros((num_raw_faces, 2), dtype=int)
        self.face_owner = np.zeros(num_raw_faces, dtype=int)
        self.face_neighbor = np.zeros(num_raw_faces, dtype=int)
        self.face_tags = [None] * num_raw_faces # Initialize list

        for i, (edge_nodes, cell_indices) in enumerate(edge_to_cells.items()):
            n1 = self.node_id_map[edge_nodes[0]]
            n2 = self.node_id_map[edge_nodes[1]]
            
            self.face_nodes[i, 0] = n1
            self.face_nodes[i, 1] = n2
            self.face_owner[i] = cell_indices[0]
            
            if len(cell_indices) > 1:
                self.face_neighbor[i] = cell_indices[1]
                self.face_tags[i] = "Internal"
            else:
                self.face_neighbor[i] = -1
                # NEW: Look up the tag from the Mesh object
                if edge_nodes in mesh.edges:
                    self.face_tags[i] = mesh.edges[edge_nodes].bc_tag
                else:
                    self.face_tags[i] = "Unknown"

        self.num_faces = num_raw_faces

    def _calculate_geometric_properties(self):
        # ... (Keep existing setup code for n1, n2, x1, y1 etc.) ...
        n1 = self.face_nodes[:, 0]
        n2 = self.face_nodes[:, 1]
        x1, y1 = self.nodes_x[n1], self.nodes_y[n1]
        x2, y2 = self.nodes_x[n2], self.nodes_y[n2]
        
        # 1. Face Centers
        self.face_centers_x = 0.5 * (x1 + x2)
        self.face_centers_y = 0.5 * (y1 + y2)

        # 2. Raw 2D Area (Length)
        dx = x2 - x1
        dy = y2 - y1
        length_2d = np.sqrt(dx**2 + dy**2)
        
        # --- AXISYMMETRIC MODIFICATION ---
        # Face Area = Length * 2 * pi * Radius_Face
        # We assume the axis of revolution is y=0
        radius_face = np.abs(self.face_centers_y)
        # Avoid zero radius at centerline to prevent zero area
        radius_face = np.maximum(radius_face, 1e-6)
        
        self.face_areas = length_2d * 2.0 * np.pi * radius_face 
        # ---------------------------------

        # 3. Normals (Same as before)
        inv_mag = 1.0 / (length_2d + 1e-12) # Note: Normalize by LENGTH, not Area
        nx = dy * inv_mag
        ny = -dx * inv_mag
        
        # Orientation check (Same as before)
        owners = self.face_owner
        ox = self.cell_centers_x[owners]
        oy = self.cell_centers_y[owners]
        dot = (self.face_centers_x - ox) * nx + (self.face_centers_y - oy) * ny
        flip_mask = dot < 0
        nx[flip_mask] *= -1.0
        ny[flip_mask] *= -1.0
        
        self.face_normals_x = nx
        self.face_normals_y = ny

        # 4. Cell Volumes
        # Pappus's Second Theorem: Volume = Area_2D * 2 * pi * Radius_Centroid
        cn = self.cell_nodes
        cx1 = self.nodes_x[cn[:, 0]]
        cy1 = self.nodes_y[cn[:, 0]]
        cx2 = self.nodes_x[cn[:, 1]]
        cy2 = self.nodes_y[cn[:, 1]]
        cx3 = self.nodes_x[cn[:, 2]]
        cy3 = self.nodes_y[cn[:, 2]]
        
        # 2D Area (Shoelace)
        term1 = cx1 * (cy2 - cy3)
        term2 = cx2 * (cy3 - cy1)
        term3 = cx3 * (cy1 - cy2)
        area_2d = 0.5 * np.abs(term1 + term2 + term3)
        
        # --- AXISYMMETRIC MODIFICATION ---
        # Radius of centroid
        radius_cell = np.abs(self.cell_centers_y)
        self.cell_volumes = area_2d * 2.0 * np.pi * radius_cell
        
        
'''



