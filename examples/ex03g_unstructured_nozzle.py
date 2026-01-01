"""
ex11_unstructured.py
--------------------
Goal: Generate a mesh for the 'ex11' Nozzle Geometry.
Fix: Dynamic Boundary Discretization to prevent slivers.
"""
import numpy as np
import matplotlib.pyplot as plt

# Project Imports
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapmesh.quality import MeshQuality
import unit_convert as cv

def get_nozzle_polygon(target_h):
    """
    Reconstructs the nozzle boundary, dynamically adjusting the 
    number of points to match 'target_h'.
    """
    # --- 1. Define Parameters ---
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

    # --- 3. Dynamic Discretization Helper ---
    def discretize_arc(center, r, start_ang, end_ang):
        # Calculate Arc Length
        angle_diff = abs(end_ang - start_ang)
        arc_len = r * angle_diff
        # Decide N based on target_h
        n = max(2, int(np.round(arc_len / target_h)))
        theta = np.linspace(start_ang, end_ang, n)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        return np.column_stack([x, y])

    def discretize_line(p1, p2):
        dist = np.linalg.norm(p2 - p1)
        n = max(2, int(np.round(dist / target_h)))
        t = np.linspace(0, 1, n)
        # Broadcasting for interpolation
        return p1 + t[:,None] * (p2 - p1)

    # --- 4. Build Curves ---
    
    # A. Inlet Vertical
    p_inlet = discretize_line(np.array([xi, 0.0]), np.array([xi, ri]))
    
    # B. Top Wall
    # 1. Inlet Arc (Center: xi, ri-rci)
    # Start: 90 deg (pi/2), End: 90 - ani
    pts_arc1 = discretize_arc(
        (xi, ri-rci), rci, 
        np.pi/2, np.pi/2 - ani
    )
    
    # 2. Converging Line
    pts_conv = discretize_line(
        np.array([xtan, rtan]), 
        np.array([xt1, rt1])
    )
    
    # 3. Throat Arc (Center: xt, rt+rct)
    # Start: 270 - ani, End: 270 + ane
    pts_arc2 = discretize_arc(
        (xt, rt+rct), rct,
        3*np.pi/2 - ani, 3*np.pi/2 + ane
    )
    
    # 4. Diverging Line
    pts_div = discretize_line(
        np.array([xt2, rt2]), 
        np.array([xe, re])
    )
    
    # C. Exit Vertical
    p_exit = discretize_line(np.array([xe, re]), np.array([xe, 0.0]))
    
    # D. Centerline
    p_center = discretize_line(np.array([xe, 0.0]), np.array([xi, 0.0]))
    
    # --- 5. Assemble (Remove duplicates at joins) ---
    full_poly = np.vstack([
        p_inlet,
        pts_arc1[1:],
        pts_conv[1:],
        pts_arc2[1:],
        pts_div[1:],
        p_exit[1:],
        p_center[1:]
    ])
    
    return full_poly, xt

def constant_sizing(x, y):
    return 0.005 

def run():
    print("--- Unstructured Nozzle (Dynamic Boundary) ---")
    
    # Define Sizing
    h_base = 0.005 # 5mm
    
    # 1. Get Boundary (Matched to h_base!)
    poly, x_throat = get_nozzle_polygon(target_h=h_base)
    
    print(f"Generated Boundary with {len(poly)} points (matches h={h_base})")
    
    # 2. Generate Mesh
    mesh = generate_unstructured_mesh(
        [poly], 
        constant_sizing,
        h_base=h_base,
        n_smooth=30
    )
    
    # 3. Quality
    MeshQuality(mesh).print_report()
    
    # 4. Visualize
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_aspect('equal')
    
    x_vals, y_vals, tris = [], [], []
    id_map = {n.id: i for i, n in enumerate(mesh.nodes.values())}
    
    for n in mesh.nodes.values():
        x_vals.append(n.x); y_vals.append(n.y)
    for c in mesh.cells.values():
        tris.append([id_map[c.n1.id], id_map[c.n2.id], id_map[c.n3.id]])
        
    ax.triplot(x_vals, y_vals, tris, 'k-', lw=0.3)
    ax.plot(poly[:,0], poly[:,1], 'r-', lw=1.5, label='Boundary')
    
    ax.set_title(f"Nozzle Mesh (Dynamic Boundary)")
    plt.show()

if __name__ == "__main__":
    run()