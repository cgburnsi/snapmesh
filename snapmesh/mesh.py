"""
snapmesh/mesh.py
----------------
The Manager. 
Links Topology (Nodes) to Geometry (Constraints).
UPDATED: Implements Node Deduplication (Topological Merging).
"""
import numpy as np
from .elements import Node, Edge, Cell
from .geometry import GeometricConstraint, TAG_CORNER

class Mesh:
    def __init__(self):
        # Topology
        self.nodes = {}       # {id: Node}
        self.edges = {}       # {id: Edge}
        self.cells = {}       # {id: Cell}
        
        # Spatial Lookup for Deduplication
        # Key: (round(x, 6), round(y, 6)) -> Node
        self._coord_map = {}
        self._merge_tol = 1e-6
        
        # Edge Uniqueness Lookup
        self._edge_lookup = {}
        
        # Geometry Registry
        self.curves = []

        # Counters
        self._next_node_id = 1
        self._next_edge_id = 1
        self._next_cell_id = 1

    def add_curve(self, curve_obj):
        if not isinstance(curve_obj, GeometricConstraint):
            raise TypeError("Object must inherit from GeometricConstraint")
        self.curves.append(curve_obj)

    def discretize_boundary(self, sizing_func):
        boundary_nodes = []
        if not self.curves: return []

        print(f"Discretizing {len(self.curves)} curves...")

        for i, curve in enumerate(self.curves):
            min_pts = 3
            if "Arc" in str(type(curve)): min_pts = 10
            
            pts, tags = curve.discretize(sizing_func, min_points=min_pts)
            
            # Smart Loop Closure:
            # If the last curve ended at X, and this curve starts at X,
            # we want to ensure we pick up the same node.
            
            # We iterate ALL points. Our new add_node will handle the merging.
            # However, we must be careful not to double-add the start point 
            # if it was already added by the previous curve's end point.
            
            # Logic: Always add the first point.
            # If it merges with previous curve's end, great.
            
            # Just add everything and let the Deduplicator handle it.
            # But we typically skip the first point of subsequent curves 
            # to avoid "logical" duplication in the list, even if IDs merge.
            
            start_k = 1 if i > 0 else 0
            
            for k in range(start_k, len(pts)):
                x, y = pts[k]
                tag = tags[k]
                
                n = self.add_node(x, y)
                n.constraint = curve
                n.is_corner = (tag == TAG_CORNER)
                boundary_nodes.append(n)
        
        # Explicitly check loop closure (Last point vs First point)
        # If the geometry is a closed loop, the last node added should merge with the first node.
        # Our add_node does this automatically based on coordinates.
        
        return boundary_nodes

    def add_node(self, x, y):
        """ Creates a node or returns an existing one if within tolerance. """
        # Quantize coordinates for fuzzy matching
        key = (round(x, 6), round(y, 6))
        
        if key in self._coord_map:
            # Check exact distance to be sure
            existing_node = self._coord_map[key]
            dist = np.sqrt((x - existing_node.x)**2 + (y - existing_node.y)**2)
            if dist < self._merge_tol:
                return existing_node
        
        # Create new
        nid = self._next_node_id
        n = Node(nid, x, y)
        self.nodes[nid] = n
        self._coord_map[key] = n
        self._next_node_id += 1
        return n

    def get_or_create_edge(self, n1, n2):
        if n1.id < n2.id:
            key = (n1.id, n2.id)
        else:
            key = (n2.id, n1.id)
            
        if key in self._edge_lookup:
            return self._edge_lookup[key]
        else:
            eid = self._next_edge_id
            e = Edge(eid, n1, n2)
            self.edges[eid] = e
            self._edge_lookup[key] = e
            self._next_edge_id += 1
            return e

    def add_cell(self, n1_id, n2_id, n3_id):
        cid = self._next_cell_id
        n1 = self.nodes[n1_id]
        n2 = self.nodes[n2_id]
        n3 = self.nodes[n3_id]
        
        e1 = self.get_or_create_edge(n1, n2)
        e2 = self.get_or_create_edge(n2, n3)
        e3 = self.get_or_create_edge(n3, n1)
        
        c = Cell(cid, n1, n2, n3, e1, e2, e3)
        self.cells[cid] = c
        self._next_cell_id += 1
        return c