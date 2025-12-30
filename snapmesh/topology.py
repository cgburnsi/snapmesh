import numpy as np

# Establish a tolerance for avoiding floating point errors in equality checks
GEOM_TOL = 1e-12

class Node:
    ''' Represents a topological vertex in the mesh with precise geometric coordinates.

    This class uses `__slots__` for memory optimization, as meshes often contain
    hundres to millions of node instances. Nodes act as the primary anchor points 
    for cells and edges. They can optionally hold a geometric `constraint`, allowing 
    them to "snap" to or slide along defined boundary curves during mesh generation
    or smoothing.  Internally, nodes coordinates are in SI units (meters) for 
    consistency across other code bases.

    Attributes:
        id (int):  Unique integer identifier for the node
        x (float): [m] The global X-coordinate
        y (float): [m] The global Y-coordinate
        constraint (object, optional): A reference to a geometric constraint 
            entity (e.g., a parametric Curve or Segment) that restricts this 
            node's position. Used to maintain shape fidelity during optimization.
    '''
    __slots__ = ['id', 'x', 'y', 'constraint'] 

    def __init__(self, nid, x, y):
        self.id = int(nid)
        self.x  = float(x)
        self.y  = float(y)
        self.constraint = None          

    def snap(self):
        ''' If Node is constrained, snap() moves the node to the closed point
            on the geometric constraint. '''
        if self.constraint:
            self.constraint.snap(self)

    def to_array(self):
        ''' Returns specific coordinates as a numpy array for calculation. '''
        return np.array([self.x, self.y], dtype=np.float64)

    def update_from_array(self, arr):
        ''' Updates coordinates from a numpy array result. '''
        self.x = float(arr[0])
        self.y = float(arr[1])

    def __repr__(self):
        return f'Node(id = {self.id:4d}: x = {self.x:10.4f}, y = {self.y:10.4f})'



class Edge:
    ''' Represents a connection between two Nodes. 
    
    Attributes:
        id (int): Unique integer identifier for the edge
        node_a (Node): Reference to the start node object.
        node_b (Node): Reference to the end node object.
        is_boundary (bool): True if this edge is on the mesh boundary.
        bc_tag (object): Optional tag for boundary conditions (e.g., WALL).
        constraint (object, optional): A reference to a geometric constraint 
            entity (e.g., a parametric Curve or Segment) that restricts this 
            edges position. Used to maintain shape fidelity during optimization.
    '''
    __slots__ = ['id', 'node_a', 'node_b', 'is_boundary', 'bc_tag', 'constraint']

    def __init__(self, eid, node_a, node_b):
        self.id = eid  
        self.node_a = node_a
        self.node_b = node_b
        
        self.is_boundary = False
        self.bc_tag = None
        self.constraint = None

    @property
    def length(self):
        ''' Calculates Euclidean length of edge on the fly (numpy vectorized). '''
        diff = self.node_b.to_array() - self.node_a.to_array()
        return np.linalg.norm(diff)

    @property
    def midpoint(self):
        ''' Returns the (x, y) midpoint as a numpy array. '''
        return 0.5 * (self.node_a.to_array() + self.node_b.to_array())

    @property
    def vector(self):
        ''' Returns the vector (B - A). '''
        return self.node_b.to_array() - self.node_a.to_array()

    def __repr__(self):
        return (f'Edge(Node A ID: {self.node_a.id:3d}, '
                f'Node B ID: {self.node_b.id:3d}, '
                f'len = {self.length:10.3f})')
    
    

class Cell:
    ''' Represents a triangular Finite Volume Cell.
    
    This class is the fundamental volumetric unit for the solver. It strictly 
    maintains a collection of 3 Nodes and 3 Edges. Crucially, it enforces 
    Counter-Clockwise (CCW) winding order upon initialization. This guarantees 
    that the signed area is positive and that edge normal vectors point 
    outwards, which is a hard requirement for flux calculations in FVM.

    Attributes:
        id (int): Unique integer identifier for the cell.
        n1, n2, n3 (Node): The three vertices of the triangle (ordered CCW).
        e1, e2, e3 (Edge): The three edges connecting the vertices.
    
    Properties:
        area (float): The signed area of the triangle (always positive due to CCW enforcement).
        center (np.ndarray): The (x, y) centroid of the triangle.
    '''
    __slots__ = ['id', 'n1', 'n2', 'n3', 'e1', 'e2', 'e3']

    def __init__(self, cid, n1, n2, n3, e1, e2, e3):
        self.id = int(cid)
        
        # --- Calculate Signed Area For Topology Checks ---        
        area_sn = Cell._calculate_signed_area(n1, n2, n3)
        
        # --- Check for Degenerate (Very Small) Cells ---
        if abs(area_sn) < GEOM_TOL:
            raise ValueError(f'Cell {cid} is degenerate (Area={area_sn:.2e}). '
                             f'Nodes: {n1.id}, {n2.id}, {n3.id} are collinear/coincident.')
            
        # --- Enforce Winding Order: Counter Clock Wise (CCW) ---
        if area_sn < -GEOM_TOL:
            # Must swap Nodes and Edges to ensure CCW winding
            n2, n3 = n3, n2
            e1, e3 = e3, e1

        # --- Store Validated Data ---
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    @staticmethod
    def _calculate_signed_area(n1, n2, n3):
        ''' Calculates 2D cross product to determine signed area.
        Result > 0 : Counter-Clockwise (CCW)
        Result < 0 : Clockwise (CW)
        '''
        return 0.5 * ((n2.x - n1.x) * (n3.y - n1.y) -
                      (n3.x - n1.x) * (n2.y - n1.y))

    @property
    def area(self):
        ''' Calculates Cell Area on the fly using the current self state. '''
        return Cell._calculate_signed_area(self.n1, self.n2, self.n3)

    @property
    def center(self):
        ''' Returns the geometric centroid (average of vertices). '''
        return (self.n1.to_array() + self.n2.to_array() + self.n3.to_array()) / 3.0

    def __repr__(self):
        return f"Cell(id={self.id}, nodes=({self.n1.id}, {self.n2.id}, {self.n3.id}))"
           

if __name__ == '__main__':
    
    a1 = Node(33, 3.2341, 1334.3)
    a2 = Node(8, 0.55, 1.4)
    a3 = Node(1, 2.2, 0.1)
    e1 = Edge(1, a1, a2)
    e2 = Edge(2, a2, a3)
    e3 = Edge(3, a3, a1)
    c1 = Cell(1, a1, a2, a3, e1, e2, e3)
    
    print(a1)
    print(a2)
    print(a3)
    print()
    print(e1)
    print(e2)
    print(e3)
    print()
    print(c1)


