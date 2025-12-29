import snapmesh as sm
import matplotlib.pyplot as plt

# Ensure you have added PointConstraint to snapmesh/__init__.py 
# or import it directly if needed:
# from snapmesh.mesh.geometry import PointConstraint

def plot_mesh(m, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        color = 'r-' if e.is_boundary else 'b-'
        lw = 1.5 if e.is_boundary else 0.5
        ax.plot([n1.x, n2.x], [n1.y, n2.y], color, lw=lw)
    ax.set_aspect('equal')
    ax.set_title(f"{title} ({len(m.cells)} cells)")
    plt.show()

def fix_corner(mesh, target_x, target_y):
    """
    Finds the node closest to (target_x, target_y) and 
    forces it to have a PointConstraint.
    """
    # 1. Find the node
    best_node = None
    min_dist = 1e-6
    
    for n in mesh.nodes.values():
        dist = (n.x - target_x)**2 + (n.y - target_y)**2
        if dist < min_dist:
            min_dist = dist
            best_node = n
            
    # 2. Lock it down
    if best_node:
        print(f"Locking corner at ({target_x}, {target_y})")
        # We manually overwrite whatever constraint was there (e.g. the Curve)
        # with a PointConstraint. This node will NEVER move now.
        best_node.constraint = sm.Point(target_x, target_y)
    else:
        print(f"Warning: No node found near corner ({target_x}, {target_y})")

if __name__ == '__main__':
    # 1. Geometry Setup
    # Updated coefficients so the curve connects (0,3) to (10,2) perfectly
    # y = 0.002x^3 - 0.03x^2 + 3
    # Start (x=0): y=3. Slope=0.
    # End   (x=10): y=2. Slope=0.
    wall_curve = sm.CubicCurve(0, 10, a=0.002, b=-0.03, c=0.0, d=3.0)
    
    inlet = sm.Line(0, 0, 0, 3)
    outlet = sm.Line(10, 2, 10, 0)
    centerline = sm.Line(10, 0, 0, 0)
    
    # Order: Inlet -> Wall -> Outlet -> Centerline
    nozzle = sm.CompositeCurve([inlet, wall_curve, outlet, centerline])

    # 2. Create Initial Mesh
    m = sm.Mesh()
    # Use coarse count to demonstrate refinement clearly
    m.add_boundary_loop(nozzle, count=12, tag="Wall")
    
    # --- MANUAL CORNER FIX ---
    # Even if the mesher logic is fuzzy, we explicitly lock the 4 corners.
    # Inlet Bottom (0,0)
    fix_corner(m, 0, 0)
    # Inlet Top (0,3)
    fix_corner(m, 0, 3)
    # Outlet Top (10,2)
    fix_corner(m, 10, 2)
    # Outlet Bottom (10,0)
    fix_corner(m, 10, 0)

    plot_mesh(m, "Level 0: Coarse (Corners Fixed)")

    # 3. Refine Level 1
    # The 'refine_global' function copies constraints.
    # So the corners will copy the PointConstraint and stay fixed,
    # while the edges copy the CurveConstraint and curve.
    m1 = sm.refine_global(m)
    plot_mesh(m1, "Level 1: Refined")

    # 4. Refine Level 2
    m2 = sm.refine_global(m1)
    plot_mesh(m2, "Level 2: Smoother")