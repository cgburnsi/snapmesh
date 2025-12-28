import snapmesh.mesh as sm_mesh  # Import the Mesh class module

def refine_global(old_mesh):
    """
    Performs 1-to-4 subdivision on the entire mesh.
    Returns a NEW Mesh object.
    """
    new_mesh = sm_mesh.Mesh()
    
    # --- 1. Copy Old Nodes ---
    # We map old_node_id -> new_node_object to keep track of connections
    old_to_new_nodes = {}
    
    for n_id, old_node in old_mesh.nodes.items():
        # Create identical node in new mesh
        new_n = new_mesh.add_node(old_node.x, old_node.y)
        
        # Copy constraints (corner nodes stay pinned to their geometry)
        new_n.constraint = old_node.constraint
        
        # Store mapping
        old_to_new_nodes[n_id] = new_n

    # --- 2. Create Midpoints for Every Edge ---
    # We need a map: (n1, n2) -> new_midpoint_node
    # This ensures neighbors share the SAME midpoint.
    edge_to_midpoint = {}

    for key, edge in old_mesh.edges.items():
        # Get the two old nodes
        n1 = old_mesh.nodes[edge.n1]
        n2 = old_mesh.nodes[edge.n2]

        # Calculate geometric midpoint
        mid_x = (n1.x + n2.x) / 2.0
        mid_y = (n1.y + n2.y) / 2.0
        
        # Create the new node
        mid_node = new_mesh.add_node(mid_x, mid_y)
        
        # CRITICAL: If the edge was on a boundary, the midpoint 
        # must attach to that geometry and SNAP.
        if edge.constraint:
            mid_node.constraint = edge.constraint
            mid_node.snap()  # Moves from linear midpoint to curved wall
            
        # If the edge had a boundary tag (e.g., "Wall"), we'll need to 
        # propogate that to the new split edges later.
        # For now, just store the midpoint.
        edge_to_midpoint[key] = mid_node

    # --- 3. Build New Cells (1 -> 4) ---
    for cell in old_mesh.cells.values():
        # Get the 3 corner nodes (in the new mesh)
        n1 = old_to_new_nodes[cell.n1]
        n2 = old_to_new_nodes[cell.n2]
        n3 = old_to_new_nodes[cell.n3]

        # Get the 3 midpoint nodes
        # Helper to safely look up edges regardless of (n1,n2) vs (n2,n1) order
        def get_mid(id_a, id_b):
            key = tuple(sorted((id_a, id_b)))
            return edge_to_midpoint[key]

        m12 = get_mid(cell.n1, cell.n2)
        m23 = get_mid(cell.n2, cell.n3)
        m31 = get_mid(cell.n3, cell.n1)

        # Add the 4 new triangles
        # 1. Corner 1
        new_mesh.add_cell(n1.id, m12.id, m31.id)
        # 2. Corner 2
        new_mesh.add_cell(m12.id, n2.id, m23.id)
        # 3. Corner 3
        new_mesh.add_cell(m31.id, m23.id, n3.id)
        # 4. Center
        new_mesh.add_cell(m12.id, m23.id, m31.id)

    # --- 4. Re-Tag Boundary Edges ---
    # (Optional but recommended) 
    # The new edges between boundary nodes and midpoints should inherit the tag.
    for key, old_edge in old_mesh.edges.items():
        if old_edge.is_boundary:
            # Reconstruct the two new segments
            n1_new = old_to_new_nodes[old_edge.n1]
            n2_new = old_to_new_nodes[old_edge.n2]
            mid    = edge_to_midpoint[key]
            
            # Segment 1: Start -> Mid
            e1 = new_mesh.tag_boundary_edge(n1_new.id, mid.id, old_edge.bc_tag)
            e1.constraint = old_edge.constraint
            
            # Segment 2: Mid -> End
            e2 = new_mesh.tag_boundary_edge(mid.id, n2_new.id, old_edge.bc_tag)
            e2.constraint = old_edge.constraint

    return new_mesh