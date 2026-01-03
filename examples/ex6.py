import numpy as np
import snapcore.units as cv
import math
import snapmesh as sm
import matplotlib.pyplot as plt

# --- 1. Helper for Units (Replacing convert.py) ---
def inches_to_meters(inches):
    return inches * 0.0254

def create_100mN_nozzle():
    
    # Specify Geometry Points
    xi  = snapcore.units.convert(0.31, 'inch', 'm')         # [m] Axial (x-axis) location of nozzle inlet
    ri  = snapcore.units.convert(2.50, 'inch', 'm')         # [m] Radial (y-axis) location of nozzle inlet
    rci = snapcore.units.convert(0.80, 'inch', 'm')         # [m] Radius of curvature of Nozzle Contraction
    rt  = snapcore.units.convert(0.80, 'inch', 'm')         # [m] Radial (y-axis) location of nozzle throat
    rct = snapcore.units.convert(0.50, 'inch', 'm')         # [m] Radius of Curvature of nozzle throat exit
    xe  = snapcore.units.convert(4.05, 'inch', 'm')         # [m] Axial (x-axis) location of nozzle exit
    
    # Specify Nozzle Angles
    ani = np.deg2rad(44.88)                     # [rad] Nozzle Contraction Angle
    ane = np.deg2rad(15.0)                      # [rad] Nozzle Expansion Angle
    
    # Inet Arc Endpoint (Tangent Point)
    
    # Create Fixed Geometry Points
    p1 = sm.Point(xi, 0)                        # Nozzle Inlet Centerline
    p2 = sm.Point(xi, ri)                       # Nozzle Inlet Wall
    
    print(p2)



# --- 2. The Geometry Builder ---
def create_nozzle_boundary():
    """
    Calculates the key geometric points based on the logic in geom1.py
    and returns a snapmesh CompositeCurve.
    """
    # -- A. Define Parameters (from geom1.py) --
    XI  = inches_to_meters(0.31)
    RI  = inches_to_meters(2.50)
    RT  = inches_to_meters(0.80)
    XE  = inches_to_meters(4.05)
    RCI = inches_to_meters(0.80)
    RCT = inches_to_meters(0.50)
    
    # Angles in Radians for math
    ANI_rad = np.deg2rad(44.88)
    ANE_rad = np.deg2rad(15.0)

    # -- B. Calculate Key Coordinates (The Math from GEOM function) --
    
    # 1. Inlet Arc End (Tangent Point)
    XTAN = XI + RCI * np.sin(ANI_rad)
    RTAN = RI + RCI * (np.cos(ANI_rad) - 1.0)
    
    # 2. Throat Entry (XT1, RT1) 
    #    (Intersection of convergent line and throat circle)
    RT1 = RT - RCT * (np.cos(ANI_rad) - 1.0)
    
    # Calculate XT1 intersection logic
    # XT1 is where the tangent line from (XTAN, RTAN) hits the throat circle
    # Slope of line = -tan(ANI)
    XT1 = XTAN + (RTAN - RT1) / np.tan(ANI_rad)
    
    # 3. Throat Location (XT)
    #    The throat circle center is at X = XT.
    #    The point (XT1, RT1) is on the circle, so we back-calculate XT.
    XT = XT1 + RCT * np.sin(ANI_rad)
    
    # 4. Throat Exit (XT2, RT2)
    XT2 = XT + RCT * np.sin(ANE_rad)
    RT2 = RT + RCT * (1.0 - np.cos(ANE_rad))
    
    # 5. Nozzle Exit (XE, RE)
    #    Project line from (XT2, RT2) with slope tan(ANE)
    RE = RT2 + (XE - XT2) * np.tan(ANE_rad)

    # -- C. Build snapmesh Objects --
    
    # Segment 1: Inlet Rounding Arc
    # Center is directly below the start point (XI, RI) by RCI
    # Start Angle: 90 deg (Top)
    # End Angle:   90 - 44.88 deg (Rotating CW)
    arc_inlet = sm.Arc(
        cx=XI, cy=RI - RCI, 
        r=RCI, 
        start_angle_deg=90.0, 
        end_angle_deg=90.0 - np.rad2deg(ANI_rad)
    )
    
    # Segment 2: Convergent Cone (Line)
    line_conv = sm.Line(XTAN, RTAN, XT1, RT1)
    
    # Segment 3: Throat Arc (Combined Upstream + Downstream)
    # Center is at (XT, RT + RCT)
    # Start Angle: 270 - ANI (Approaching from left/up)
    # End Angle:   270 + ANE (Exiting to right/up)
    # Note: 270 deg is straight down (6 o'clock), which is the throat itself.
    arc_throat = sm.Arc(
        cx=XT, cy=RT + RCT,
        r=RCT,
        start_angle_deg=270.0 - np.rad2deg(ANI_rad),
        end_angle_deg=270.0 + np.rad2deg(ANE_rad)
    )
    
    # Segment 4: Divergent Cone (Line)
    line_div = sm.Line(XT2, RT2, XE, RE)
    
    # Segment 5: Enclosure (Exit -> Centerline -> Inlet)
    line_exit  = sm.Line(XE, RE, XE, 0.0)
    line_center= sm.Line(XE, 0.0, XI, 0.0) # From Exit back to Inlet
    line_inlet = sm.Line(XI, 0.0, XI, RI)  # Up to start point

    # -- D. Combine --
    # Note: We group the Wall separate from the enclosure if we want to tag them differently later.
    # For now, let's make one big loop.
    return sm.CompositeCurve([
        arc_inlet, 
        line_conv, 
        arc_throat, 
        line_div, 
        line_exit, 
        line_center, 
        line_inlet
    ])

# --- 3. Main Execution ---
if __name__ == '__main__':
    # 1. Get the Math-Defined Geometry
    nozzle_geom = create_nozzle_boundary()
    
    # 2. Create Mesh
    m = sm.Mesh()
    
    # 3. Add Loop
    # Note: We use a higher count to capture the arcs nicely
    m.add_boundary_loop(nozzle_geom, count=50, tag="Nozzle")
    
    # 4. Explicitly Fix Key Corners (Optional but good for robustness)
    #    Inlet Start (XI, 0)
    XI_val = inches_to_meters(0.31)
    XE_val = inches_to_meters(4.05)
    sm.refine_global  # Ensure imported
    
    # Helper to fix corners (copy from ex5)
    def fix_corner(mesh, tx, ty):
        best_n, min_d = None, 1e-6
        for n in mesh.nodes.values():
            d = (n.x-tx)**2 + (n.y-ty)**2
            if d < min_d: min_d, best_n = d, n
        if best_n: best_n.constraint = sm.Point(tx, ty)

    # Lock the 4 corners of the domain
    fix_corner(m, XI_val, 0.0) # Inlet Center
    fix_corner(m, XI_val, inches_to_meters(2.50)) # Inlet Top
    fix_corner(m, XE_val, 0.0) # Exit Center
    # We don't fix Exit Top because it's a derived value (RE), 
    # but the logic naturally puts a node there.

    # 5. Refine
    print("Refining Level 1...")
    m1 = sm.refine_global(m)
    print("Refining Level 2...")
    m2 = sm.refine_global(m1)

    # 6. Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for e in m2.edges.values():
        n1 = m2.nodes[e.n1]
        n2 = m2.nodes[e.n2]
        color = 'k-' if e.is_boundary else 'b-'
        lw = 1.0 if e.is_boundary else 0.5
        ax.plot([n1.x, n2.x], [n1.y, n2.y], color, lw=lw)
    
    ax.set_aspect('equal')
    ax.set_title("Nozzle Generated from geom1.py Logic")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    create_100mN_nozzle()