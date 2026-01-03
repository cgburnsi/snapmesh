import numpy as np
import snapcore.units as cv
import math
import snapmesh as sm
import matplotlib.pyplot as plt



def create_100mN_nozzle(): 
    # Specify Geometry Points and nozzle angles
    xi  = snapcore.units.convert(0.31, 'inch', 'm')         # [m] Axial (x-axis) location of nozzle inlet
    ri  = snapcore.units.convert(2.50, 'inch', 'm')         # [m] Radial (y-axis) location of nozzle inlet
    rci = snapcore.units.convert(0.80, 'inch', 'm')         # [m] Radius of curvature of Nozzle Contraction
    rt  = snapcore.units.convert(0.80, 'inch', 'm')         # [m] Radial (y-axis) location of nozzle throat
    rct = snapcore.units.convert(0.50, 'inch', 'm')         # [m] Radius of Curvature of nozzle throat exit
    xe  = snapcore.units.convert(4.05, 'inch', 'm')         # [m] Axial (x-axis) location of nozzle exit
    ani = np.deg2rad(44.88)                     # [rad] Nozzle Contraction Angle
    ane = np.deg2rad(15.0)                      # [rad] Nozzle Expansion Angle
    
    # --- Calculate the critical point locations ---
    # Inlet Arc Endpoint (Tangent Point)
    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    
    # Throat Entry Point (xt1, rt1) - Intersection of convergent line and throat circle
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan-rt1) / np.tan(ani)  # slope of line = -tan(ani)

    # Throat axial location (rt already specified above)
    xt = xt1 + rct * np.sin(ani)

    # Throat Exit Location (intersection of throat exit circle and line)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
        
    # Nozzle Exit (xe, re) (project line from (XT2, RT2) with slope tan(ANE))
    re = rt2 + (xe-xt2) * np.tan(ane)
    
    # --- Determine curves that define the equations for the wall, inlet, outlet, and centerline ---
    # Segment 1: Inlet Rounding Arc (center is directly below (xi, ri), Angle (Start=90, End = 90-deg(ani))
    arc_inlet = sm.Arc(cx=xi, cy=(ri-rci), r=rci, 
                       start_angle=np.deg2rad(90.0), 
                       end_angle=np.deg2rad(90.0) - ani)
    # Segment 2: Convergent Cone (Line)
    line_conv = sm.Line(xtan, rtan, xt1, rt1)
    # Segment 3: Throat Arc (Combined Upstream + Downstream)
    # Center is at (XT, RT + RCT)
    # Start Angle: 270 - ANI (Approaching from left/up)
    # End Angle:   270 + ANE (Exiting to right/up)
    # Note: 270 deg is straight down (6 o'clock), which is the throat itself.
    arc_throat = sm.Arc(cx=xt, cy=(rt+rct), r=rct,
                        start_angle=np.deg2rad(270.0) - ani,
                        end_angle=np.deg2rad(270.0) + ane)
    # Segment 4: Divergent Cone
    line_div = sm.Line(xt2, rt2, xe, re)
    # Segment 5: Enclosure (Exit -> Centerline -> Inlet)
    line_exit   = sm.Line(xe, re, xe, 0.0)
    line_center = sm.Line(xe, 0.0, xi, 0.0) # From Exit back to Inlet
    line_inlet  = sm.Line(xi, 0.0, xi, ri)  # Up to start point
    
    # --- Create a dictionary of of the points calculated in this function for validation and plotting ---
    points = {'Inlet Center': (xi, 0.0),
              'Inlet Wall':   (xi, ri),
              'Inlet Tangent':(xtan, rtan),
              'Throat Entry': (xt1, rt1),
              'Throat Center':(xt, rt), 
              'Throat Exit':  (xt2, rt2),
              'Nozzle Exit':  (xe, re),
              'Exit Center':  (xe, 0.0)}
    
    # --- Create the geometry (It's a big single loop right now, but will change when we enforce tags)
    geom = sm.CompositeCurve([arc_inlet, 
        line_conv, 
        arc_throat, 
        line_div, 
        line_exit, 
        line_center, 
        line_inlet
    ])
    
    return geom, points




# --- 3. Main Execution ---
if __name__ == '__main__':
    
    # 1. Get Geometry AND Points
    nozzle_geom, pts = create_100mN_nozzle()
    
    # 2. Create Mesh
    m = sm.Mesh()
    m.add_boundary_loop(nozzle_geom, count=50, tag="Wall")

    # 3. Lock Fixed Points
    # Use the returned 'pts' dictionary to lock nodes automatically
    for name, (px, py) in pts.items():
        best_n, min_d = None, 1e-6
        for n in m.nodes.values():
            dist = (n.x - px)**2 + (n.y - py)**2
            if dist < min_d:
                min_d = dist
                best_n = n
        
        if best_n:
            # Explicitly lock the node
            best_n.constraint = sm.Point(px, py)

    # 4. Refine
    #print("Refining Level 1...")
    #m = sm.refine_global(m)
    #print("Refining Level 2...")
    #m = sm.refine_global(m)

    # 5. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Mesh
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        color = 'k' if e.is_boundary else 'b-'
        lw = 1.0 if e.is_boundary else 0.5
        ax.plot([n1.x, n2.x], [n1.y, n2.y], color, lw=lw)

    # Plot Verification Points (Red Dots)
    for name, (px, py) in pts.items():
        ax.plot(px, py, 'ro', markersize=6)

    ax.set_aspect('equal')
    ax.set_title("100mN Nozzle Mesh (Radians in API)")
    plt.grid(True, alpha=0.3)
    plt.show()