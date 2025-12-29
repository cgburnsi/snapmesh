import unit_convert as cv
import numpy as np
import snapmesh as sm
from snapmesh.transfinite import generate_structured_mesh
import matplotlib.pyplot as plt

def create_structured_nozzle():
    # --- 1. Define Parameters (Same as before) ---
    xi  = cv.convert(0.31, 'inch', 'm')
    ri  = cv.convert(2.50, 'inch', 'm')
    rci = cv.convert(0.80, 'inch', 'm')
    rt  = cv.convert(0.80, 'inch', 'm')
    rct = cv.convert(0.50, 'inch', 'm')
    xe  = cv.convert(4.05, 'inch', 'm')
    ani = np.deg2rad(44.88)
    ane = np.deg2rad(15.0)

    # --- 2. Calculate Key Points ---
    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt = xt1 + rct * np.sin(ani)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)

    # --- 3. Build The 4 Logical Sides ---
    
    # LEFT: The Inlet Line
    # Note: Direction matters! We normally go Counter-Clockwise.
    # Left side goes Bottom -> Top (Axis -> Wall)
    left_curve = sm.Line(xi, 0.0, xi, ri)
    
    # RIGHT: The Exit Line
    # Right side goes Bottom -> Top? No, structured mesh logic usually expects
    # Left(Bottom->Top) and Right(Bottom->Top) so u=0 connects to u=1.
    # Let's define Right side as (Exit Axis -> Exit Wall)
    right_curve = sm.Line(xe, 0.0, xe, re)

    # BOTTOM: Centerline
    # Goes Left -> Right (Inlet -> Exit)
    bottom_curve = sm.Line(xi, 0.0, xe, 0.0)

    # TOP: The Complex Wall
    # Goes Left -> Right (Inlet -> Exit)
    # We combine ALL the wall segments into one CompositeCurve
    arc_inlet = sm.Arc(cx=xi, cy=(ri-rci), r=rci, 
                       start_angle=np.deg2rad(90.0), 
                       end_angle=np.deg2rad(90.0) - ani)
    line_conv = sm.Line(xtan, rtan, xt1, rt1)
    arc_throat = sm.Arc(cx=xt, cy=(rt+rct), r=rct,
                        start_angle=np.deg2rad(270.0) - ani,
                        end_angle=np.deg2rad(270.0) + ane)
    line_div = sm.Line(xt2, rt2, xe, re)
    
    # This is the "Top Side"
    top_curve = sm.CompositeCurve([arc_inlet, line_conv, arc_throat, line_div])

    # --- 4. Generate Mesh ---
    # 60 nodes streamwise (ni), 15 nodes radial (nj)
    mesh = generate_structured_mesh(bottom_curve, top_curve, left_curve, right_curve, ni=10, nj=5)
    
    return mesh

if __name__ == '__main__':
    # Generate
    m = create_structured_nozzle()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        color = 'k-' if e.is_boundary else 'b-'
        alpha = 1.0 if e.is_boundary else 0.3
        ax.plot([n1.x, n2.x], [n1.y, n2.y], color, alpha=alpha, lw=0.5)
        
    ax.set_aspect('equal')
    ax.set_title("Structured Mesh (Transfinite Interpolation)")
    plt.show()