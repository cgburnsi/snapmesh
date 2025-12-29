import math
import snapmesh as sm
from enum import Enum

# --- Entities ---
class BCTag(Enum):
    WALL = 1
    INLET = 2
    OUTLET = 3

class Node:
    def __init__(self, nid, x, y):
        self.id = nid
        self.x = x
        self.y = y
        self.constraint = None  # Holds the geometry object (Circle, Line, etc.)

    def snap(self):
        """If constrained, move to the closest point on the geometry."""
        if self.constraint:
            self.constraint.snap(self)

class Edge:
    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.is_boundary = False
        self.bc_tag = None
        self.constraint = None

class Cell:
    def __init__(self, cid, n1, n2, n3):
        self.id = cid
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

# --- The Mesh Class ---
class Mesh:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cells = {}
        self.node_counter = 0
        self.cell_counter = 0

    def add_node(self, x, y):
        """Creates a node and returns the object."""
        self.node_counter += 1
        n = Node(self.node_counter, x, y)
        self.nodes[n.id] = n
        return n

    def add_cell(self, n1_id, n2_id, n3_id):
        """Creates a triangle cell."""
        self.cell_counter += 1
        c = Cell(self.cell_counter, n1_id, n2_id, n3_id)
        self.cells[c.id] = c
        
        # Register edges implicitly so we can tag them later
        self._register_edge(n1_id, n2_id)
        self._register_edge(n2_id, n3_id)
        self._register_edge(n3_id, n1_id)
        return c

    def _register_edge(self, n1, n2):
        """Internal helper to track edges."""
        key = tuple(sorted((n1, n2)))
        if key not in self.edges:
            self.edges[key] = Edge(n1, n2)
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