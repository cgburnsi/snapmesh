import math
import numpy as np

from enum import Enum
from .elements import Node, Edge, Cell

# --- Entities ---
class BCTag(Enum):
    WALL = 1
    INLET = 2
    OUTLET = 3




# --- The Mesh Class ---
class Mesh:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cells = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.cell_counter = 0

    def add_node(self, x, y):
        """Creates a node and returns the object."""
        self.node_counter += 1
        n = Node(self.node_counter, x, y)
        self.nodes[n.id] = n
        return n

# In mesh.py inside Mesh class

    def add_cell(self, n1_id, n2_id, n3_id):
        self.cell_counter += 1
        
        # 1. Get Node Objects
        n1 = self.nodes[n1_id]
        n2 = self.nodes[n2_id]
        n3 = self.nodes[n3_id]
        
        # 2. Get Edge Objects
        e1 = self._register_edge(n1_id, n2_id)
        e2 = self._register_edge(n2_id, n3_id)
        e3 = self._register_edge(n3_id, n1_id)
        
        # 3. Create the Triangle Cell
        c = Cell(self.cell_counter, n1, n2, n3, e1, e2, e3)
        self.cells[c.id] = c
        
        return c

# In mesh.py

    def _register_edge(self, n1_id, n2_id):
        """
        Internal helper to track edges. 
        Ensures every unique edge gets a unique ID for FVM.
        """
        key = tuple(sorted((n1_id, n2_id)))
        
        if key not in self.edges:
            # 1. Retrieve the actual Node objects
            node_obj_a = self.nodes[n1_id]
            node_obj_b = self.nodes[n2_id]
            
            # 2. Increment the ID counter
            self.edge_counter += 1
            new_eid = self.edge_counter
            
            # 3. Create the Edge with the MANDATORY ID
            # This matches your new __init__(self, eid, node_a, node_b)
            self.edges[key] = Edge(new_eid, node_obj_a, node_obj_b)
            
        return self.edges[key]

    def tag_boundary_edge(self, n1_id, n2_id, tag):
        """Marks an existing edge as a boundary."""
        key = tuple(sorted((n1_id, n2_id)))
        if key in self.edges:
            e = self.edges[key]
            e.is_boundary = True
            e.bc_tag = tag
            return e
        else:
            # Edge doesn't exist yet (mesh might be incomplete), create it
            e = self._register_edge(n1_id, n2_id)
            e.is_boundary = True
            e.bc_tag = tag
            return e

    def add_boundary_loop(self, curve, count, tag=None):
        created_nodes = []
        
        # --- STRATEGY A: COMPOSITE CURVE (Smart Corner Handling) ---
        if hasattr(curve, "segments"):
            segs = curve.segments
            num_segs = len(segs)
            
            # Distribute the total 'count' among the segments.
            # (Simple approach: equal nodes per segment. Can be improved later)
            nodes_per_seg = max(1, count // num_segs)
            
            for seg_idx, seg in enumerate(segs):
                # Generate nodes for this segment
                # We go from 0 to N-1 (inclusive) to avoid duplicating the endpoint,
                # because the endpoint of this segment is the start of the next one.
                for j in range(nodes_per_seg):
                    t = j / float(nodes_per_seg)
                    x, y = seg.evaluate(t)
                    n = self.add_node(x, y)
                    
                    # Constraint Logic:
                    # 1. The start of a segment (j=0) is a CORNER. Fix it.
                    if j == 0:
                        n.constraint = None # Fixed/Point constraint
                    else:
                        n.constraint = seg  # Sliding constraint
                        
                    created_nodes.append(n)

        # --- STRATEGY B: SIMPLE CURVE (Circle, etc) ---
        else:
            for i in range(count):
                t = i / float(count)
                x, y = curve.evaluate(t)
                n = self.add_node(x, y)
                n.constraint = curve
                created_nodes.append(n)
            
        # --- CONNECT EDGES ---
        total_nodes = len(created_nodes)
        for i in range(total_nodes):
            n_curr = created_nodes[i]
            n_next = created_nodes[(i + 1) % total_nodes]
            
            edge = self.tag_boundary_edge(n_curr.id, n_next.id, tag)
            
            # For the edge constraint, we need to know which segment it belongs to.
            # Since we built 'created_nodes' in segment order, we can infer it.
            if hasattr(curve, "segments"):
                # Map global index 'i' back to segment index
                nodes_per_seg = max(1, count // len(curve.segments))
                seg_idx = (i // nodes_per_seg) % len(curve.segments)
                edge.constraint = curve.segments[seg_idx]
            else:
                edge.constraint = curve
            
        return created_nodes