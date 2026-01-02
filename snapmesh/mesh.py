"""
snapmesh/mesh.py
----------------
The Manager. 
Links Topology (Nodes) to Geometry (Constraints).
UPDATED: Fixes add_cell() to automatically manage Edges.
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
        
        # Edge Uniqueness Lookup: (min_id, max_id) -> Edge Object
        self._edge_lookup = {}
        
        # Geometry Registry
        self.curves = []      # List of GeometricConstraint objects

        # Counters
        self._next_node_id = 1
        self._next_edge_id = 1
        self._next_cell_id = 1

    def add_curve(self, curve_obj):
        """ Registers a geometric boundary curve. """
        if not isinstance(curve_obj, GeometricConstraint):
            raise TypeError("Object must inherit from GeometricConstraint")
        self.curves.append(curve_obj)

    def discretize_boundary(self, sizing_func):
        """
        Generates boundary Nodes based on the registered Curves.
        """
        boundary_nodes = []
        
        if not self.curves:
            print("Warning: No curves registered in Mesh.")
            return []

        print(f"Discretizing {len(self.curves)} curves...")

        for i, curve in enumerate(self.curves):
            # 1. Ask curve for points
            # Arcs generally need higher fidelity
            min_pts = 3
            if "Arc" in str(type(curve)): min_pts = 10
            
            pts, tags = curve.discretize(sizing_func, min_points=min_pts)
            
            # 2. Convert to Nodes
            # Chain logic: Skip first point if not the first curve
            start_idx = 1 if i > 0 else 0
            
            for k in range(start_idx, len(pts)):
                x, y = pts[k]
                tag = tags[k]
                
                n = self.add_node(x, y)
                
                # Link Node to Geometry
                n.constraint = curve
                
                # Tag for generator
                n.is_corner = (tag == TAG_CORNER)
                
                boundary_nodes.append(n)
        
        return boundary_nodes

    # --- Element Management ---
    def add_node(self, x, y):
        nid = self._next_node_id
        n = Node(nid, x, y)
        self.nodes[nid] = n
        self._next_node_id += 1
        return n

    def get_or_create_edge(self, n1, n2):
        """ 
        Returns an existing edge between n1 and n2, or creates a new one.
        Ensures topological uniqueness (1-2 is same as 2-1).
        """
        # Sort IDs to create a unique key for undirected edge
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
        
        # Automatically find or create the required edges
        e1 = self.get_or_create_edge(n1, n2)
        e2 = self.get_or_create_edge(n2, n3)
        e3 = self.get_or_create_edge(n3, n1)
        
        # Initialize Cell with full topology
        c = Cell(cid, n1, n2, n3, e1, e2, e3)
        self.cells[cid] = c
        self._next_cell_id += 1
        return c