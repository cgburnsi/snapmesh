import numpy as np
import math

class UnstructuredGrid:
    def __init__(self, mesh):
        self.num_cells = len(mesh.cells)
        self.num_nodes = len(mesh.nodes)
        
        # --- 1. Geometry Data ---
        self.nodes_x = np.zeros(self.num_nodes)
        self.nodes_y = np.zeros(self.num_nodes)
        
        # Map node ID -> Array Index (0 to N-1)
        # Also create a reverse map to look up cell IDs later if needed
        self.node_id_map = {} 
        self.cell_id_map = {} # Map cell.id -> 0..N index
        
        for i, (nid, n) in enumerate(mesh.nodes.items()):
            self.nodes_x[i] = n.x
            self.nodes_y[i] = n.y
            self.node_id_map[nid] = i

        # --- 2. Topology Data (Faces) ---
        self.faces = []
        self._build_topology(mesh)
        
        # --- 3. Geometric Properties ---
        self._calculate_geometric_properties()

    def _build_topology(self, mesh):
        # 1. Identify Edges shared by Cells
        edge_to_cells = {}

        # Map cell IDs to 0..N indices
        for i, cid in enumerate(mesh.cells.keys()):
            self.cell_id_map[cid] = i

        for cid, cell in mesh.cells.items():
            c_idx = self.cell_id_map[cid]
            
            # The 3 edges of the triangle
            # Sort node IDs so (n1,n2) is same key as (n2,n1)
            es = [
                tuple(sorted((cell.n1, cell.n2))),
                tuple(sorted((cell.n2, cell.n3))),
                tuple(sorted((cell.n3, cell.n1)))
            ]
            
            for e in es:
                if e not in edge_to_cells:
                    edge_to_cells[e] = []
                edge_to_cells[e].append(c_idx)

        # 2. Convert to Owner-Neighbor Lists
        self.face_nodes = [] 
        self.face_owner = [] 
        self.face_neighbor = [] 
        
        for edge_nodes, cell_indices in edge_to_cells.items():
            n1 = self.node_id_map[edge_nodes[0]]
            n2 = self.node_id_map[edge_nodes[1]]
            
            self.face_nodes.append([n1, n2])
            
            # Owner is just the first cell found
            self.face_owner.append(cell_indices[0])
            
            # Neighbor
            if len(cell_indices) > 1:
                self.face_neighbor.append(cell_indices[1])
            else:
                self.face_neighbor.append(-1) # Boundary

        # Convert to numpy
        self.face_nodes = np.array(self.face_nodes, dtype=int)
        self.face_owner = np.array(self.face_owner, dtype=int)
        self.face_neighbor = np.array(self.face_neighbor, dtype=int)
        self.num_faces = len(self.face_nodes)

    def _calculate_geometric_properties(self):
        # Initialize arrays
        self.cell_centers_x = np.zeros(self.num_cells)
        self.cell_centers_y = np.zeros(self.num_cells)
        self.face_centers_x = np.zeros(self.num_faces)
        self.face_centers_y = np.zeros(self.num_faces)

        # --- Cell Centroids ---
        # For triangles, centroid is just average of 3 nodes
        # We assume face_owner/neighbor indices map to 0..num_cells-1 correctly
        # But we need to calculate them based on the actual mesh.cells
        # Let's iterate the node arrays using face data? 
        # Easier: iterate the original mesh cells again using our map.
        
        # We need a reverse lookup from index -> cell object to get nodes easily?
        # Actually, let's just do it broadly:
        # (This is a simplified centroid calc)
        
        counts = np.zeros(self.num_cells)
        
        # We can iterate faces to sum up node positions for cells? 
        # No, easier to just calculate face centers first.
        
        n1_indices = self.face_nodes[:, 0]
        n2_indices = self.face_nodes[:, 1]
        
        x1 = self.nodes_x[n1_indices]
        y1 = self.nodes_y[n1_indices]
        x2 = self.nodes_x[n2_indices]
        y2 = self.nodes_y[n2_indices]
        
        self.face_centers_x = 0.5 * (x1 + x2)
        self.face_centers_y = 0.5 * (y1 + y2)
        
        # Compute Cell Centroids (Average of its face centers? Or nodes?)
        # For a triangle, average of vertices is centroid.
        # Let's rely on the fact that we know the node coords.
        
        # Temporary accumulation
        cx = np.zeros(self.num_cells)
        cy = np.zeros(self.num_cells)
        
        # We iterate faces and add face center contribution to owner/neighbor?
        # That's messy. Let's just iterate the faces and add the specific node coords?
        # Actually, let's just cheat and assume we can access the mesh object logic later.
        # But for strictly using 'grid' data:
        
        # Let's use the face-to-node data to rebuild cell-to-node data?
        # A bit circular. Let's just iterate the faces.
        # Every internal face contributes to 2 cells. Boundary to 1.
        # This is getting complicated.
        
        # SIMPLE WAY: Just iterate the faces.
        # For every face, add its midpoint to the owner/neighbor accumulation? 
        # No, that's not the centroid.
        
        # CORRECT WAY for this class:
        # We'll just init them to 0 and sum face centers / 3? No.
        
        # Let's just pass the mesh logic in __init__ properly.
        pass 
        
        # Actually, let's calculate cell centroids *during* init while we still have the mesh object
        # See updated __init__ below.

    # Redefine __init__ to do centroids simply
    def __init__(self, mesh):
        self.num_cells = len(mesh.cells)
        self.num_nodes = len(mesh.nodes)
        
        self.nodes_x = np.zeros(self.num_nodes)
        self.nodes_y = np.zeros(self.num_nodes)
        self.node_id_map = {} 
        self.cell_id_map = {} 

        # 1. Fill Nodes
        for i, (nid, n) in enumerate(mesh.nodes.items()):
            self.nodes_x[i] = n.x
            self.nodes_y[i] = n.y
            self.node_id_map[nid] = i
            
        # 2. Calculate Cell Centroids immediately from Mesh
        self.cell_centers_x = np.zeros(self.num_cells)
        self.cell_centers_y = np.zeros(self.num_cells)
        
        for i, (cid, cell) in enumerate(mesh.cells.items()):
            self.cell_id_map[cid] = i
            n1 = mesh.nodes[cell.n1]
            n2 = mesh.nodes[cell.n2]
            n3 = mesh.nodes[cell.n3]
            
            self.cell_centers_x[i] = (n1.x + n2.x + n3.x) / 3.0
            self.cell_centers_y[i] = (n1.y + n2.y + n3.y) / 3.0
            
        # 3. Build Topology
        self._build_topology(mesh)
        
        # 4. Face Centers
        n1 = self.face_nodes[:, 0]
        n2 = self.face_nodes[:, 1]
        self.face_centers_x = 0.5 * (self.nodes_x[n1] + self.nodes_x[n2])
        self.face_centers_y = 0.5 * (self.nodes_y[n1] + self.nodes_y[n2])