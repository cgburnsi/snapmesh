"""
ex04c_geometry_objects.py
-------------------------
Goal: Demonstrate the restored Object-Oriented Geometry system.
      UPDATED: Uses 'min_points' to guarantee arc fidelity.
"""
import numpy as np
import matplotlib.pyplot as plt

# Project Imports
import snapmesh.geometry as geom
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapmesh.quality import MeshQuality
import unit_convert as cv

def create_nozzle_curves():
    # ... (Same Parameters as before) ...
    xi  = cv.convert(0.31, 'inch', 'm')
    ri  = cv.convert(2.50, 'inch', 'm')
    rci = cv.convert(0.80, 'inch', 'm')
    rt  = cv.convert(0.80, 'inch', 'm')
    rct = cv.convert(0.50, 'inch', 'm')
    xe  = cv.convert(4.05, 'inch', 'm')
    ani = np.deg2rad(44.88)
    ane = np.deg2rad(15.0)

    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt = xt1 + rct * np.sin(ani)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)

    # --- Object Definition ---
    inlet = geom.LineSegment([xi, 0.0], [xi, ri])
    
    arc1 = geom.Arc((xi, ri-rci), rci, np.pi/2, np.pi/2 - ani)
    conv = geom.LineSegment(arc1.evaluate(1.0), [xt1, rt1])
    
    arc2 = geom.Arc((xt, rt+rct), rct, 3*np.pi/2 - ani, 3*np.pi/2 + ane)
    div  = geom.LineSegment(arc2.evaluate(1.0), [xe, re])
    
    exit_line = geom.LineSegment([xe, re], [xe, 0.0])
    center    = geom.LineSegment([xe, 0.0], [xi, 0.0])
    
    return [inlet, arc1, conv, arc2, div, exit_line, center], xt, rt

def create_sizing_func(x_throat, r_throat):
    h_min = 0.0015
    h_max = 0.008
    width = 3.0 * r_throat 
    def sizing(x, y):
        dist = np.abs(x - x_throat)
        ratio = np.clip(dist / width, 0.0, 1.0)
        factor = ratio * ratio * (3 - 2 * ratio)
        return h_min + factor * (h_max - h_min)
    return sizing

def run():
    print("--- Geometry Object Test (Fidelity Check) ---")
    
    curves, xt, rt = create_nozzle_curves()
    sizing = create_sizing_func(xt, rt)
    
    poly_points = []
    poly_tags = []
    
    for c in curves:
        # --- SMART DISCRETIZATION ---
        # If it's an Arc, force high resolution (15 points min)
        # If it's a Line, let it be adaptive (2 points min)
        if isinstance(c, geom.Arc):
            min_pts = 15
        else:
            min_pts = 2
            
        pts, tags = c.discretize_adaptive(sizing, min_points=min_pts)
        
        if len(poly_points) > 0:
            poly_points.append(pts[1:])
            poly_tags.append(tags[1:])
        else:
            poly_points.append(pts)
            poly_tags.append(tags)
            
    full_poly = np.vstack(poly_points)
    full_tags = np.concatenate(poly_tags)
    
    print(f"Generated Boundary: {len(full_poly)} points.")
    
    # Mesh it
    mesh = generate_unstructured_mesh(
        [(full_poly, full_tags)], 
        sizing,
        h_base=0.008,
        n_smooth=30
    )
    
    MeshQuality(mesh).print_report()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_aspect('equal')
    
    x_vals = [n.x for n in mesh.nodes.values()]
    y_vals = [n.y for n in mesh.nodes.values()]
    tris = [[c.n1.id, c.n2.id, c.n3.id] for c in mesh.cells.values()]
    id_map = {uid: i for i, uid in enumerate(mesh.nodes.keys())}
    tris_idx = [[id_map[u], id_map[v], id_map[w]] for u,v,w in tris]
    
    ax.triplot(x_vals, y_vals, tris_idx, 'k-', lw=0.3)
    
    # Prove the geometry match
    t = np.linspace(0, 1, 50)
    for c in curves:
        pts = np.array([c.evaluate(ti) for ti in t])
        ax.plot(pts[:,0], pts[:,1], 'r-', lw=2, alpha=0.6)
        
    plt.title("Nozzle Mesh (With Arc Fidelity)")
    plt.show()

if __name__ == "__main__":
    run()