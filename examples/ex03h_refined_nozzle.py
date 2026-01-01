"""
ex03h_refined_nozzle.py
-----------------------
Goal: Generate a nozzle mesh with AUTOMATIC LOCAL REFINEMENT.
      - Tuned for better resolution retention at the throat.
"""
import numpy as np
import matplotlib.pyplot as plt

# Project Imports
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapmesh.quality import MeshQuality
import unit_convert as cv

# --- 1. Sizing Function Setup (TUNED) ---
def create_nozzle_sizing(x_throat, r_throat):
    """
    Returns a function f(x,y) -> h.
    Physics-based sizing with wider influence to prevent resolution drop.
    """
    # Parameters (Tuned for higher resolution)
    h_min = 0.0015 # 1.5mm at throat (was 2.0mm)
    h_max = 0.008  # 8.0mm at inlet/outlet (was 10.0mm)
    
    # Widen the influence zone to keep mesh fine longer
    # Use 3.0 radii instead of 2.0
    width = 3.0 * r_throat 
    
    def sizing(x, y):
        dist = np.abs(x - x_throat)
        
        # Use a smooth cosine or cubic blend instead of linear
        # This makes the transition gentler
        ratio = np.clip(dist / width, 0.0, 1.0)
        
        # Cubic smoothstep: 3x^2 - 2x^3 (Classic easing function)
        factor = ratio * ratio * (3 - 2 * ratio)
        
        return h_min + factor * (h_max - h_min)
        
    return sizing

# --- 2. Adaptive Geometry Discretization (ROBUST) ---
def discretize_adaptive(curve_func, t_start, t_end, length, sizing_func):
    """
    Walks along a parametric curve and places points based on local sizing.
    Includes a safety factor to ensure we don't step too big.
    """
    points = []
    t = t_start
    
    points.append(curve_func(t))
    
    direction = 1.0 if t_end > t_start else -1.0
    
    # Safety: Limit max iterations to prevent infinite loops
    max_iter = 10000
    it = 0
    
    while (t < t_end) if direction > 0 else (t > t_end):
        p_current = curve_func(t)
        
        # Get local h req
        h_req = sizing_func(p_current[0], p_current[1])
        
        # CONSERVATIVE STEPPING:
        # Only take 90% of the required step to ensure we don't undersample curves
        dt = (0.9 * h_req / length) * direction
        
        t += dt
        it += 1
        
        if (t >= t_end) if direction > 0 else (t <= t_end):
            break
            
        points.append(curve_func(t))
        if it > max_iter: break
        
    points.append(curve_func(t_end))
    return np.array(points)

def get_refined_boundary(sizing_func):
    """
    Reconstructs the nozzle boundary.
    """
    # --- Define Parameters ---
    xi  = cv.convert(0.31, 'inch', 'm')
    ri  = cv.convert(2.50, 'inch', 'm')
    rci = cv.convert(0.80, 'inch', 'm')
    rt  = cv.convert(0.80, 'inch', 'm')
    rct = cv.convert(0.50, 'inch', 'm')
    xe  = cv.convert(4.05, 'inch', 'm')
    
    ani = np.deg2rad(44.88)
    ane = np.deg2rad(15.0)

    # --- Calculate Key Points ---
    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt = xt1 + rct * np.sin(ani) # Throat X
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)

    # --- Parametric Curves Helpers ---
    def line_p(p1, p2): 
        return lambda t: p1 + t * (p2 - p1)
    
    # --- Build Segments ---
    
    # A. Inlet Vertical
    p_start, p_end = np.array([xi, 0.0]), np.array([xi, ri])
    L = np.linalg.norm(p_end - p_start)
    pts_inlet = discretize_adaptive(line_p(p_start, p_end), 0.0, 1.0, L, sizing_func)
    
    # B. Top Wall - 1. Inlet Arc
    L = rci * ani
    center1 = (xi, ri-rci)
    func_arc1 = lambda t: np.array([center1[0] + rci*np.cos(t), center1[1] + rci*np.sin(t)])
    pts_arc1 = discretize_adaptive(func_arc1, np.pi/2, np.pi/2 - ani, L, sizing_func)
    
    # Top Wall - 2. Conv Line
    p1, p2 = np.array([xtan, rtan]), np.array([xt1, rt1])
    L = np.linalg.norm(p2 - p1)
    pts_conv = discretize_adaptive(line_p(p1, p2), 0.0, 1.0, L, sizing_func)
    
    # Top Wall - 3. Throat Arc
    center2 = (xt, rt+rct)
    angle_start = 3*np.pi/2 - ani
    angle_end   = 3*np.pi/2 + ane
    L = rct * (ane + ani)
    func_arc2 = lambda t: np.array([center2[0] + rct*np.cos(t), center2[1] + rct*np.sin(t)])
    pts_arc2 = discretize_adaptive(func_arc2, angle_start, angle_end, L, sizing_func)
    
    # Top Wall - 4. Div Line
    p1, p2 = np.array([xt2, rt2]), np.array([xe, re])
    L = np.linalg.norm(p2 - p1)
    pts_div = discretize_adaptive(line_p(p1, p2), 0.0, 1.0, L, sizing_func)
    
    # C. Exit Vertical
    p1, p2 = np.array([xe, re]), np.array([xe, 0.0])
    L = np.linalg.norm(p2 - p1)
    pts_exit = discretize_adaptive(line_p(p1, p2), 0.0, 1.0, L, sizing_func)
    
    # D. Centerline
    p1, p2 = np.array([xe, 0.0]), np.array([xi, 0.0])
    L = np.linalg.norm(p2 - p1)
    pts_center = discretize_adaptive(line_p(p1, p2), 0.0, 1.0, L, sizing_func)
    
    # --- Stitch ---
    full_poly = np.vstack([
        pts_inlet,
        pts_arc1[1:],
        pts_conv[1:],
        pts_arc2[1:],
        pts_div[1:],
        pts_exit[1:],
        pts_center[1:]
    ])
    
    return full_poly, xt, rt

def run():
    print("--- Refined Nozzle Generation (Tuned) ---")
    
    # 1. Define Sizing Logic
    xi  = cv.convert(0.31, 'inch', 'm')
    rci = cv.convert(0.80, 'inch', 'm')
    rt  = cv.convert(0.80, 'inch', 'm')
    rct = cv.convert(0.50, 'inch', 'm')
    ani = np.deg2rad(44.88)
    
    xtan = xi + rci * np.sin(ani)
    rtan = cv.convert(2.50, 'inch', 'm') + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt_approx = xt1 + rct * np.sin(ani)
    
    # Create Function
    size_func = create_nozzle_sizing(xt_approx, rt)
    
    # 2. Generate Boundary
    print("Discretizing Boundary...")
    poly, xt, _ = get_refined_boundary(size_func)
    print(f"  -> {len(poly)} boundary points generated.")
    
    # 3. Generate Mesh
    print("Meshing Interior...")
    mesh = generate_unstructured_mesh(
        [poly],
        size_func,
        h_base=0.008, # Use the coarser size as base
        n_smooth=30
    )
    
    # 4. Quality
    MeshQuality(mesh).print_report()
    
    # 5. Visualize
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
    ax.axvline(xt, color='b', linestyle='--', alpha=0.5, label='Throat')
    
    ax.set_title("Refined Nozzle Mesh (Tuned Transition)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()