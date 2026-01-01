"""
mesh.py
-------
The central container for all mesh data.
Manages IDs, storage, and relationships between topology and geometry.
"""
import numpy as np

from enum import Enum
from .elements import Node, Edge, Cell
from .geometry import LineSegment  # Updated from 'Line'

# --- Entities ---
class BCTag(Enum):
    WALL = 1
    INLET = 2
    OUTLET = 3





# --- The Mesh Class ---
class Mesh:
    def __init__(self):
        # Data Storage (Dicts for O(1) lookup)
        self.nodes = {}       # {id: Node}
        self.edges = {}       # {id: Edge}
        self.cells = {}       # {id: Cell}
        self.constraints = {} # {id: GeometryObject}

        # ID Counters (Auto-increment)
        # We use internal counters to ensure every entity has a unique ID
        self._next_node_id = 1
        self._next_edge_id = 1
        self._next_cell_id = 1
        self._next_geom_id = 1

    # --- Geometry Management ---
    def add_line(self, p1, p2):
        """ 
        Creates a LineSegment constraint and registers it. 
        Returns the new Geometry ID (int).
        """
        # 1. Create the Geometry Object
        geom = LineSegment(p1, p2)
        
        # 2. Assign ID and Store
        gid = self._next_geom_id
        self.constraints[gid] = geom
        
        self._next_geom_id += 1
        return gid

    # --- Topology Management ---
    def add_node(self, x, y, constraint_id=None):
        """ 
        Factory method to create a Node. 
        If constraint_id is provided, links the Node to that Geometry and snaps it.
        """
        nid = self._next_node_id
        node = Node(nid, x, y)
        
        # Link to Geometry (The "Option B" Magic)
        if constraint_id is not None:
            if constraint_id not in self.constraints:
                raise KeyError(f"Geometry ID {constraint_id} not found in Mesh.")
            
            # Retrieve the object and assign it to the node
            geom_obj = self.constraints[constraint_id]
            node.constraint = geom_obj
            
            # Snap immediately to ensure the node starts in a valid state
            node.snap()

        self.nodes[nid] = node
        self._next_node_id += 1
        return node

    def add_edge(self, n1, n2):
        """ 
        Creates or retrieves an edge between two node objects. 
        """
        # Use a sorted tuple key so (n1, n2) is the same as (n2, n1)
        key = tuple(sorted((n1.id, n2.id)))
        
        if key not in self.edges:
            eid = self._next_edge_id
            edge = Edge(eid, n1, n2)
            self.edges[key] = edge # Store by tuple key for lookup
            
            self._next_edge_id += 1
            
        return self.edges[key]
    
    def add_circle(self, center, radius):
        """ 
        Creates a Circle constraint and registers it.
        """
        # Lazy import to avoid circular dependency
        from .geometry import Circle
        
        geom = Circle(center, radius)
        
        gid = self._next_geom_id
        self.constraints[gid] = geom
        
        self._next_geom_id += 1
        return gid
    

    def add_cell(self, n1_id, n2_id, n3_id):
        """ Creates a cell from 3 Node IDs. Automatically manages Edges. """
        cid = self._next_cell_id
        
        # 1. Retrieve Node Objects
        n1 = self.nodes[n1_id]
        n2 = self.nodes[n2_id]
        n3 = self.nodes[n3_id]
        
        # 2. Create/Retrieve Edge Objects
        e1 = self.add_edge(n1, n2)
        e2 = self.add_edge(n2, n3)
        e3 = self.add_edge(n3, n1)
        
        # 3. Create Cell (Topology check happens inside Cell.__init__)
        cell = Cell(cid, n1, n2, n3, e1, e2, e3)
        self.cells[cid] = cell
        
        self._next_cell_id += 1
        return cell

    def tag_boundary_edge(self, n1_id, n2_id, tag):
        """Marks an existing edge as a boundary."""
        n1 = self.nodes[n1_id]
        n2 = self.nodes[n2_id]
        edge = self.add_edge(n1, n2)
        
        edge.is_boundary = True
        edge.bc_tag = tag
        return edge

    def add_boundary_loop(self, geom_id, count, bc_tag=None):
        """
        Generates a CLOSED loop of nodes/edges along a parametric curve.
        Useful for Circles, Ellipses, or closed splines.
        
        Args:
            geom_id (int): ID of the geometry constraint.
            count (int): Number of nodes/edges to generate.
            bc_tag (BCTag, optional): Tag to apply to the generated edges.
        
        Returns:
            list[Node]: The created nodes.
        """
        # 1. Retrieve Geometry
        if geom_id not in self.constraints:
            raise KeyError(f"Geometry ID {geom_id} not found.")
        
        curve = self.constraints[geom_id]
        
        # Safety Check: Can we mesh this?
        if not hasattr(curve, 'evaluate'):
             raise TypeError(f"Geometry ID {geom_id} ({type(curve).__name__}) is not a ParametricCurve.")

        nodes = []
        
        # 2. Generate Nodes (0 to 2*pi)
        # We iterate 0..count-1. We do NOT create a node at t=1.0 because 
        # that is the same geometric point as t=0.0 in a closed loop.
        for i in range(count):
            t = i / float(count)
            
            # Get mathematical position
            pos = curve.evaluate(t)
            
            # Create Node (Mesh will automatically snap it to geom_id)
            n = self.add_node(pos[0], pos[1], constraint_id=geom_id)
            nodes.append(n)
            
        # 3. Generate Edges (Connect i -> i+1, and wrap around)
        for i in range(count):
            n1 = nodes[i]
            n2 = nodes[(i + 1) % count] # Modulo operator wraps last node to first
            
            # Create/Tag Edge
            self.tag_boundary_edge(n1.id, n2.id, bc_tag)
            
        return nodes


    def merge_duplicate_nodes(self, tolerance=1e-9):
        """
        Finds nodes that are geometrically identical and merges them.
        Essential for stitching multi-block meshes or closing loops.
        """
        # 1. Map locations to Nodes
        # We use a spatial key to find duplicates efficiently
        unique_map = {} # "x_y" -> Node
        replacements = {} # old_id -> new_node_id
        
        nodes_to_remove = []
        
        # Snapshot of current nodes
        all_nodes = list(self.nodes.values())
        
        for n in all_nodes:
            # Create a string key for precision grouping
            # (In a production code, we'd use a KD-Tree, but this works for <100k nodes)
            key = f"{n.x:.9f}_{n.y:.9f}"
            
            if key in unique_map:
                # Found a duplicate! The one already in the map is the 'Keeper'
                keeper = unique_map[key]
                replacements[n.id] = keeper
                nodes_to_remove.append(n.id)
            else:
                # This is a new unique location
                unique_map[key] = n
                
        # 2. Re-wire Cells to point to the Keeper nodes
        if not replacements:
            return 0
            
        for cell in self.cells.values():
            # If this cell uses a node that is being removed, swap it!
            if cell.n1.id in replacements: cell.n1 = replacements[cell.n1.id]
            if cell.n2.id in replacements: cell.n2 = replacements[cell.n2.id]
            if cell.n3.id in replacements: cell.n3 = replacements[cell.n3.id]
            
        # 3. Remove the dead nodes
        for nid in nodes_to_remove:
            del self.nodes[nid]
            
        return len(nodes_to_remove)
















"""

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
    
    
"""