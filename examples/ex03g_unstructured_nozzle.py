"""
ex03g_unstructured_nozzle.py
----------------------------
Goal: Generate a mesh for the Nozzle Geometry.
UPDATED: Uses the Refactored Mesh & Geometry Architecture.
         (No more manual point generation!)
"""
import numpy as np
import matplotlib.pyplot as plt

# Project Imports
from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment, Arc
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapmesh.quality import MeshQuality
import unit_convert as cv

def create_nozzle_mesh(h_target):
    """
    Builds the Mesh container and defines the Geometry constraints.
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

    # --- 2. Calculate Key Geometric Points ---
    # (These are just the start/end coordinates for our objects)
    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt = xt1 + rct * np.sin(ani)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)

    # --- 3. Build the Mesh Manager ---
    mesh = Mesh()
    
    # --- 4. Define & Register Geometry Objects ---
    # A. Inlet (Vertical Line)
    inlet = LineSegment([xi, 0.0], [xi, ri])
    mesh.add_curve(inlet)
    
    # B. Inlet Arc (The "Corner" 1)
    # Note: We use the end of previous to ensure water-tightness
    arc1 = Arc((xi, ri-rci), rci, np.pi/2, np.pi/2 - ani)
    mesh.add_curve(arc1)
    
    # C. Converging Section (Straight Line)
    conv = LineSegment(arc1.evaluate(1.0), [xt1, rt1])
    mesh.add_curve(conv)
    
    # D. Throat Arc (The "Corner" 2)
    arc2 = Arc((xt, rt+rct), rct, 3*np.pi/2 - ani, 3*np.pi/2 + ane)
    mesh.add_curve(arc2)
    
    # E. Diverging Section (Straight Line)
    div = LineSegment(arc2.evaluate(1.0), [xe, re])
    mesh.add_curve(div)
    
    # F. Exit (Vertical Line)
    exit_line = LineSegment([xe, re], [xe, 0.0])
    mesh.add_curve(exit_line)
    
    # G. Centerline (Horizontal Line)
    center = LineSegment([xe, 0.0], [xi, 0.0])
    mesh.add_curve(center)
    
    # --- 5. Generate Boundary Nodes ---
    # We use a constant sizing function for this example
    def sizing(x, y):
        return h_target
        
    print("Discretizing Boundary on Geometry...")
    mesh.discretize_boundary(sizing)
    
    return mesh

def run():
    print("--- Unstructured Nozzle (Refactored Object-Oriented) ---")
    
    # Define Sizing
    h_base = 0.002 # 5mm
    
    # 1. Setup Mesh & Geometry
    mesh = create_nozzle_mesh(h_target=h_base)
    
    print(f"Generated Boundary with {len(mesh.nodes)} nodes.")
    
    # 2. Generate Interior Mesh
    # Note: We pass the MESH object, not a polygon list!
    generate_unstructured_mesh(
        mesh, 
        lambda x, y: h_base, # Constant sizing
        h_base=h_base,
        n_smooth=30
    )
    
    # 3. Quality Check
    MeshQuality(mesh).print_report()
    
    # 4. Visualize
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_aspect('equal')
    
    # Extract data for plotting
    x_vals = []
    y_vals = []
    # Map Node IDs to array indices for matplotlib
    id_map = {uid: i for i, uid in enumerate(mesh.nodes.keys())}
    
    for n in mesh.nodes.values():
        x_vals.append(n.x)
        y_vals.append(n.y)
        
    tris = []
    for c in mesh.cells.values():
        tris.append([id_map[c.n1.id], id_map[c.n2.id], id_map[c.n3.id]])
        
    ax.triplot(x_vals, y_vals, tris, 'k-', lw=0.5)
    ax.plot(x_vals, y_vals, 'k.', markersize=2)
    
    plt.title(f"Unstructured Nozzle (h={h_base}m)")
    plt.xlabel("Axial Position (m)")
    plt.ylabel("Radial Position (m)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()