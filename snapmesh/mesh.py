import math
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
        """
        Discretizes a parametric curve into 'count' segments.
        Generates nodes and boundary edges automatically.
        """
        created_nodes = []
        
        # 1. Create Nodes along the curve
        for i in range(count):
            # t goes from 0.0 to 1.0 (exclusive of 1.0 for loops)
            t = i / float(count)
            
            x, y = curve.evaluate(t)
            n = self.add_node(x, y)
            
            # Attach the curve so they stick to it later!
            n.constraint = curve 
            created_nodes.append(n)
            
        # 2. Connect Edges in a loop
        for i in range(count):
            n_curr = created_nodes[i]
            n_next = created_nodes[(i + 1) % count] # Wrap back to start
            
            edge = self.tag_boundary_edge(n_curr.id, n_next.id, tag)
            edge.constraint = curve
            
        return created_nodes