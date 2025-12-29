import snapmesh.mesh as sm_mesh
from snapmesh.geometry import Line

def generate_structured_mesh(bottom_curve, top_curve, left_curve, right_curve, ni, nj):
    """
    Generates a structured triangular mesh inside a 4-sided region.
    
    Parameters:
      ni: Number of nodes along the Bottom/Top direction (streamwise)
      nj: Number of nodes along the Left/Right direction (radial)
    """
    mesh = sm_mesh.Mesh()
    
    # Store node IDs in a 2D grid so we can connect them later
    # grid[i][j] will hold the Node object
    grid = [[None for _ in range(nj)] for _ in range(ni)]
    
    # --- 1. Evaluate Boundaries ---
    # We pre-calculate the positions of nodes along the 4 edges.
    
    # Bottom & Top (Streamwise, i=0..ni-1)
    bottom_nodes = []
    top_nodes = []
    for i in range(ni):
        u = i / float(ni - 1)
        bottom_nodes.append(bottom_curve.evaluate(u))
        top_nodes.append(top_curve.evaluate(u))
        
    # Left & Right (Radial, j=0..nj-1)
    left_nodes = []
    right_nodes = []
    for j in range(nj):
        v = j / float(nj - 1)
        left_nodes.append(left_curve.evaluate(v))
        right_nodes.append(right_curve.evaluate(v))

    # --- 2. Generate Internal Nodes (Transfinite Interpolation) ---
    for i in range(ni):
        for j in range(nj):
            u = i / float(ni - 1)
            v = j / float(nj - 1)
            
            # The 4 boundary points relevant to this (u, v) location
            xb, yb = bottom_nodes[i] # Bottom
            xt, yt = top_nodes[i]    # Top
            xl, yl = left_nodes[j]   # Left
            xr, yr = right_nodes[j]  # Right
            
            # Corner points (for the bilinear correction term)
            x00, y00 = left_nodes[0]    # Bottom-Left
            x10, y10 = right_nodes[0]   # Bottom-Right
            x01, y01 = left_nodes[-1]   # Top-Left
            x11, y11 = right_nodes[-1]  # Top-Right
            
            # Gordon-Hall Formula:
            # P(u,v) = (1-v)*Bottom(u) + v*Top(u) + (1-u)*Left(v) + u*Right(v) 
            #          - [ Bilinear Interpolation of Corners ]
            
            # X coordinate
            term1_x = (1-v)*xb + v*xt
            term2_x = (1-u)*xl + u*xr
            bilinear_x = (1-u)*(1-v)*x00 + u*(1-v)*x10 + (1-u)*v*x01 + u*v*x11
            x = term1_x + term2_x - bilinear_x
            
            # Y coordinate
            term1_y = (1-v)*yb + v*yt
            term2_y = (1-u)*yl + u*yr
            bilinear_y = (1-u)*(1-v)*y00 + u*(1-v)*y10 + (1-u)*v*y01 + u*v*y11
            y = term1_y + term2_y - bilinear_y
            
            # Create the node
            n = mesh.add_node(x, y)
            grid[i][j] = n
            
            # --- Constraints ---
            # If on boundary, lock it to the curve logic so refinement works later.
            if j == 0: n.constraint = bottom_curve
            elif j == nj-1: n.constraint = top_curve
            elif i == 0: n.constraint = left_curve
            elif i == ni-1: n.constraint = right_curve
            
            # Corners (Explicit Fixed Points)
            if i==0 and j==0: n.constraint = None # Bottom-Left (Fixed)
            if i==ni-1 and j==0: n.constraint = None # Bottom-Right
            if i==0 and j==nj-1: n.constraint = None # Top-Left
            if i==ni-1 and j==nj-1: n.constraint = None # Top-Right

    # --- 3. Connect Nodes into Triangles ---
    # We split each "square" (i, i+1, j, j+1) into 2 triangles.
    for i in range(ni - 1):
        for j in range(nj - 1):
            n1 = grid[i][j]
            n2 = grid[i+1][j]
            n3 = grid[i+1][j+1]
            n4 = grid[i][j+1]
            
            # Split Quad into 2 Triangles
            # Tri 1: (i,j) -> (i+1,j) -> (i+1,j+1)
            mesh.add_cell(n1.id, n2.id, n3.id)
            
            # Tri 2: (i,j) -> (i+1,j+1) -> (i,j+1)
            mesh.add_cell(n1.id, n3.id, n4.id)
            
            # Tag Boundaries
            if j == 0: mesh.tag_boundary_edge(n1.id, n2.id, "Bottom")
            if j == nj-2: mesh.tag_boundary_edge(n4.id, n3.id, "Top") # Note order
            if i == 0: mesh.tag_boundary_edge(n4.id, n1.id, "Left")
            if i == ni-2: mesh.tag_boundary_edge(n2.id, n3.id, "Right")

    return mesh