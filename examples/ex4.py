import snapmesh as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Building Mathematical Nozzle...")

    # --- Define Dimensions ---
    length = 10.0
    r_inlet = 3.0
    r_throat = 1.0
    r_outlet = 2.0
    
    # --- 1. The Math Curve (Top Wall) ---
    # We want a curve that goes from (0, 3) down to (5, 1) then up to (10, 2).
    # I calculated these coefficients for a smooth cubic fit:
    # y = 0.008x^3 - 0.12x^2 + 0.2x + 3
    wall_curve = sm.CubicCurve(0, length, a=0.008, b=-0.12, c=0.2, d=3.0)

    # --- 2. The Straight Edges ---
    # Inlet: (0, 0) up to (0, 3)
    inlet = sm.Line(0, 0, 0, r_inlet)
    
    # Outlet: (10, 2) down to (10, 0)
    outlet = sm.Line(length, r_outlet, length, 0)
    
    # Centerline: (10, 0) back to (0, 0)
    centerline = sm.Line(length, 0, 0, 0)

    # --- 3. Combine into a Closed Loop ---
    # Order: Inlet -> Wall -> Outlet -> Centerline
    nozzle = sm.CompositeCurve([inlet, wall_curve, outlet, centerline])

    # --- 4. Mesh ---
    m = sm.Mesh()
    
    # Use higher resolution (count=60) to capture the curve nicely
    m.add_boundary_loop(nozzle, count=60, tag="NozzleWall")

    # --- 5. Plot ---
    fig, ax = plt.subplots(figsize=(10, 4))
    
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        ax.plot([n1.x, n2.x], [n1.y, n2.y], 'b-o', markersize=3)

    ax.set_aspect('equal')
    ax.set_title("Mathematically Defined Nozzle")
    plt.grid(True, alpha=0.3)
    plt.show()