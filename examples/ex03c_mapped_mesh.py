"""
ex03c_mapped_mesh.py
--------------------
Goal: Generate a high-quality mesh for a non-rectangular shape (Trapezoid)
using Transfinite Interpolation (Mapped Meshing).
"""
import numpy as np
import matplotlib.pyplot as plt
from snapmesh.mesh import Mesh
from snapmesh.quality import MeshQuality

def transfinite_map(u, v, c1, c2, c3, c4):
    """
    Maps logical coordinates (u, v) in [0,1] to Physical Space (x, y).
    Uses Bilinear Interpolation between 4 corner points.
    
    Corners must be in order: Bottom-Left, Bottom-Right, Top-Right, Top-Left.
    """
    # Basis functions for bilinear patch
    # phi1 = (1-u)(1-v)  -> Weight for c1
    # phi2 = u(1-v)      -> Weight for c2
    # phi3 = uv          -> Weight for c3
    # phi4 = (1-u)v      -> Weight for c4
    
    phi1 = (1 - u) * (1 - v)
    phi2 = u * (1 - v)
    phi3 = u * v
    phi4 = (1 - u) * v
    
    x = phi1*c1[0] + phi2*c2[0] + phi3*c3[0] + phi4*c4[0]
    y = phi1*c1[1] + phi2*c2[1] + phi3*c3[1] + phi4*c4[1]
    
    return x, y

def create_mapped_mesh(c1, c2, c3, c4, nx, ny):
    """
    Generates a structured mesh mapped into a generic quadrangle.
    """
    mesh = Mesh()
    # 2D array to store Node IDs for connectivity
    node_ids = np.zeros((ny + 1, nx + 1), dtype=int)
    
    print(f"Mapping Grid ({nx}x{ny}) into Quad...")
    
    # 1. Create Nodes (Mapped)
    for j in range(ny + 1):
        for i in range(nx + 1):
            # Normalized coordinates (0.0 to 1.0)
            u = i / nx
            v = j / ny
            
            # Map to physical space
            px, py = transfinite_map(u, v, c1, c2, c3, c4)
            
            n = mesh.add_node(px, py)
            node_ids[j, i] = n.id

    # 2. Create Cells (Same topology as rectangle!)
    for j in range(ny):
        for i in range(nx):
            id1 = node_ids[j,   i]
            id2 = node_ids[j,   i+1]
            id3 = node_ids[j+1, i]
            id4 = node_ids[j+1, i+1]
            
            # Split quad into 2 triangles
            mesh.add_cell(id1, id2, id4)
            mesh.add_cell(id1, id4, id3)
            
    return mesh

def run():
    print("--- Mapped Mesh Test (Nozzle Section) ---")
    
    # Define a Trapezoid (Like a diverging nozzle)
    # C4 (0, 1)        C3 (2, 2)  <-- Expansion
    #  +--------------+
    #  |              |
    #  +--------------+
    # C1 (0, 0)        C2 (2, -1) <-- Expansion
    
    c1 = (0.0,  0.0)
    c2 = (2.0, -1.0) 
    c3 = (2.0,  2.0) 
    c4 = (0.0,  1.0)
    
    # Generate (15x10 grid)
    mesh = create_mapped_mesh(c1, c2, c3, c4, nx=15, ny=10)
    
    # Check Quality
    inspector = MeshQuality(mesh)
    inspector.print_report()
    inspector.plot_histograms()
    
    # Visualize Geometry
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    
    x_vals = []
    y_vals = []
    triangles = []
    
    # Map IDs to indices for plotting
    id_map = {n.id: i for i, n in enumerate(mesh.nodes.values())}
    
    for n in mesh.nodes.values():
        x_vals.append(n.x)
        y_vals.append(n.y)
        
    for c in mesh.cells.values():
        triangles.append([id_map[c.n1.id], id_map[c.n2.id], id_map[c.n3.id]])
        
    ax.triplot(x_vals, y_vals, triangles, 'k-', lw=0.5)
    
    # Draw Boundary
    bx = [c1[0], c2[0], c3[0], c4[0], c1[0]]
    by = [c1[1], c2[1], c3[1], c4[1], c1[1]]
    ax.plot(bx, by, 'r-', lw=2, label="Boundary")
    
    ax.set_title("Mapped Mesh (Trapezoid)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()