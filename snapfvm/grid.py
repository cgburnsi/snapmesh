import numpy as np
import math

class UnstructuredGrid:
    def __init__(self, mesh):
        """
        Converts a snapmesh.Mesh object into an FVM-ready UnstructuredGrid.
        
        This process "compiles" the mesh:
        1. Flattens dictionary storage into efficient Numpy arrays.
        2. Computes Face-based connectivity (Owner/Neighbor).
        3. Pre-calculates geometric metrics (Normals, Volumes, Areas).
        """
        self.num_cells = len(mesh.cells)
        self.num_nodes = len(mesh.nodes)
        
        # --- 1. Flatten Node Data ---
        # Convert node objects to flat coordinate arrays for speed
        self.nodes_x = np.zeros(self.num_nodes)
        self.nodes_y = np.zeros(self.num_nodes)
        
        # Map: Mesh Node ID -> Array Index (0 to N-1)
        self.node_id_map = {} 
        for i, (nid, n) in enumerate(mesh.nodes.items()):
            self.nodes_x[i] = n.x
            self.nodes_y[i] = n.y
            self.node_id_map[nid] = i

        # --- 2. Flatten Cell Data ---
        # Map: Mesh Cell ID -> Array Index (0 to N-1)
        self.cell_id_map = {}
        
        # Store cell->node connectivity (needed for Volume calculation)
        self.cell_nodes = np.zeros((self.num_cells, 3), dtype=int)
        
        # Pre-calculate Cell Centroids
        self.cell_centers_x = np.zeros(self.num_cells)
        self.cell_centers_y = np.zeros(self.num_cells)

        for i, (cid, cell) in enumerate(mesh.cells.items()):
            self.cell_id_map[cid] = i
            
            # Map IDs to indices
            n1_idx = self.node_id_map[cell.n1]
            n2_idx = self.node_id_map[cell.n2]
            n3_idx = self.node_id_map[cell.n3]
            
            self.cell_nodes[i, 0] = n1_idx
            self.cell_nodes[i, 1] = n2_idx
            self.cell_nodes[i, 2] = n3_idx
            
            # Centroid = Average of vertices
            self.cell_centers_x[i] = (self.nodes_x[n1_idx] + self.nodes_x[n2_idx] + self.nodes_x[n3_idx]) / 3.0
            self.cell_centers_y[i] = (self.nodes_y[n1_idx] + self.nodes_y[n2_idx] + self.nodes_y[n3_idx]) / 3.0

        # --- 3. Build Connectivity (Faces) ---
        self.faces = []
        self._build_topology(mesh)
        
        # --- 4. Calculate Metrics (Normals, Volumes) ---
        self._calculate_geometric_properties()

    def _build_topology(self, mesh):
        """
        Identifies unique faces and determines Owner/Neighbor connectivity.
        """
        # Intermediate dictionary to find shared edges
        # Key: tuple(sorted_node_indices) -> Value: List of [owner_cell_idx, neighbor_cell_idx]
        edge_to_cells = {}

        for cid, cell in mesh.cells.items():
            c_idx = self.cell_id_map[cid]
            
            # The 3 edges of the triangle
            # Sort node IDs so edge (A,B) is identical to edge (B,A)
            es = [
                tuple(sorted((cell.n1, cell.n2))),
                tuple(sorted((cell.n2, cell.n3))),
                tuple(sorted((cell.n3, cell.n1)))
            ]
            
            for e in es:
                if e not in edge_to_cells:
                    edge_to_cells[e] = []
                edge_to_cells[e].append(c_idx)

        # Initialize Arrays
        num_raw_faces = len(edge_to_cells)
        self.face_nodes = np.zeros((num_raw_faces, 2), dtype=int)
        self.face_owner = np.zeros(num_raw_faces, dtype=int)
        self.face_neighbor = np.zeros(num_raw_faces, dtype=int)
        
        # Fill Arrays
        for i, (edge_nodes, cell_indices) in enumerate(edge_to_cells.items()):
            n1 = self.node_id_map[edge_nodes[0]]
            n2 = self.node_id_map[edge_nodes[1]]
            
            self.face_nodes[i, 0] = n1
            self.face_nodes[i, 1] = n2
            
            # Owner is the first cell found
            self.face_owner[i] = cell_indices[0]
            
            # Neighbor logic
            if len(cell_indices) > 1:
                self.face_neighbor[i] = cell_indices[1] # Internal Face
            else:
                self.face_neighbor[i] = -1              # Boundary Face

        self.num_faces = num_raw_faces

    def _calculate_geometric_properties(self):
        """
        Calculates Face Normals, Face Areas (Lengths), and Cell Volumes (Areas).
        Enforces that normals point from Owner -> Neighbor.
        """
        # --- 1. Face Geometrics ---
        n1 = self.face_nodes[:, 0]
        n2 = self.face_nodes[:, 1]
        
        x1, y1 = self.nodes_x[n1], self.nodes_y[n1]
        x2, y2 = self.nodes_x[n2], self.nodes_y[n2]

        # Midpoints (Face Centers)
        self.face_centers_x = 0.5 * (x1 + x2)
        self.face_centers_y = 0.5 * (y1 + y2)

        # Edge Vector (dx, dy)
        dx = x2 - x1
        dy = y2 - y1
        
        # Face Area (Length in 2D)
        self.face_areas = np.sqrt(dx**2 + dy**2)

        # Raw Normals (Rotate 90 deg: dy, -dx)
        # We normalize them to unit vectors
        inv_mag = 1.0 / (self.face_areas + 1e-12)
        nx = dy * inv_mag
        ny = -dx * inv_mag

        # --- ORIENTATION CHECK (Crucial for FVM) ---
        # The normal must point OUT of the Owner cell.
        # Vector d = FaceCenter - OwnerCenter
        owners = self.face_owner
        ox = self.cell_centers_x[owners]
        oy = self.cell_centers_y[owners]
        
        d_x = self.face_centers_x - ox
        d_y = self.face_centers_y - oy
        
        # Dot Product
        dot = d_x * nx + d_y * ny
        
        # If dot < 0, normal is pointing inward. Flip it.
        flip_mask = dot < 0
        nx[flip_mask] *= -1.0
        ny[flip_mask] *= -1.0

        self.face_normals_x = nx
        self.face_normals_y = ny

        # --- 2. Cell Volumes (Triangle Area) ---
        cn = self.cell_nodes
        x1 = self.nodes_x[cn[:, 0]]
        y1 = self.nodes_y[cn[:, 0]]
        x2 = self.nodes_x[cn[:, 1]]
        y2 = self.nodes_y[cn[:, 1]]
        x3 = self.nodes_x[cn[:, 2]]
        y3 = self.nodes_y[cn[:, 2]]
        
        # Shoelace formula (unsigned area)
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        term1 = x1 * (y2 - y3)
        term2 = x2 * (y3 - y1)
        term3 = x3 * (y1 - y2)
        
        self.cell_volumes = 0.5 * np.abs(term1 + term2 + term3)