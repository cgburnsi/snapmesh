"""
ex03f_square_hole.py
--------------------
Goal: Test Unstructured Generation with Internal Boundaries (Holes).
Geometry: A 2x2 Square with a Radius=0.5 circular hole in the center.
"""
import numpy as np
import matplotlib.pyplot as plt
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapmesh.quality import MeshQuality

def get_square_poly():
    # 2x2 Square centered at origin (CCW)
    return [
        (-1.0, -1.0),
        ( 1.0, -1.0),
        ( 1.0,  1.0),
        (-1.0,  1.0)
    ]

def get_circle_poly(r=0.5, n=30):
    # Circle (Clockwise for Hole)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Reverse to make it Clockwise
    t = t[::-1] 
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])

def variable_sizing(x, y):
    """
    Sizing function:
    Small elements near the hole (h=0.05), larger away (h=0.2).
    """
    dist = np.sqrt(x**2 + y**2)
    # Linear ramp based on distance from center
    # dist=0.5 -> h=0.05
    # dist=1.5 -> h=0.2
    h = 0.05 + 0.15 * (dist - 0.5)
    return np.clip(h, 0.05, 0.2)

def run():
    print("--- Square with Hole Test ---")
    
    # 1. Define Loops
    outer = get_square_poly()
    inner = get_circle_poly(r=0.5, n=40)
    
    # Pass LIST of loops [Outer, Inner]
    boundaries = [outer, inner]
    
    # 2. Generate
    mesh = generate_unstructured_mesh(
        boundaries, 
        variable_sizing, 
        h_base=0.1, 
        n_smooth=30
    )
    
    print(f"Generated {len(mesh.nodes)} nodes, {len(mesh.cells)} cells.")
    
    # 3. Quality Check
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
        
    ax.triplot(x_vals, y_vals, tris, 'k-', lw=0.5)
    
    # Plot Boundaries (Red)
    # Close loops for plotting
    p_out = np.vstack([outer, outer[0]])
    p_in  = np.vstack([inner, inner[0]])
    ax.plot(p_out[:,0], p_out[:,1], 'r-', lw=2)
    ax.plot(p_in[:,0],  p_in[:,1],  'r-', lw=2)
    
    ax.set_title("Square with Hole (Variable Sizing)")
    plt.show()

if __name__ == "__main__":
    run()