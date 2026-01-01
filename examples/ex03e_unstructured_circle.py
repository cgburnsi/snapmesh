"""
ex03e_unstructured_circle.py
----------------------------
Goal: Verify the new Unstructured Generator on a Circle.
"""
import numpy as np
import matplotlib.pyplot as plt
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapmesh.quality import MeshQuality

def circle_boundary(r=1.0, n=40):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])

def constant_sizing(x, y):
    return 0.15 # Target size

def run():
    print("--- Unstructured Circle Test ---")
    
    # 1. Define Boundary (Open list of points)
    poly = circle_boundary(1.0, 40)
    
    # 2. Generate Mesh
    # (The mesher handles the wrap-around internally)
    mesh = generate_unstructured_mesh(poly, constant_sizing, h_base=0.15, n_smooth=30)
    
    # 3. Check Quality
    inspector = MeshQuality(mesh)
    inspector.print_report()
    
    # 4. Visualize
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    
    x_vals, y_vals, tris = [], [], []
    id_map = {n.id: i for i, n in enumerate(mesh.nodes.values())}
    
    for n in mesh.nodes.values():
        x_vals.append(n.x); y_vals.append(n.y)
    for c in mesh.cells.values():
        tris.append([id_map[c.n1.id], id_map[c.n2.id], id_map[c.n3.id]])
        
    # Plot Mesh (Black)
    ax.triplot(x_vals, y_vals, tris, 'k-', lw=0.5)
    
    # --- PLOTTING FIX ---
    # Append the first point to the end so the red line closes the loop
    poly_plot = np.vstack([poly, poly[0]])
    ax.plot(poly_plot[:,0], poly_plot[:,1], 'r-', lw=2, label='Boundary')
    
    ax.set_title("Unstructured Circle (Relaxed)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()