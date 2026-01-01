"""
ex03a_structured_grid.py
------------------------
Goal: Build a mesh using algebraic loops (Structured Generation).
This proves that if we give the Mesh class good data, it builds a good mesh.
"""
import numpy as np
import matplotlib.pyplot as plt
from snapmesh.mesh import Mesh

def create_structured_grid(width, height, nx, ny):
    """
    Generates a rectangular mesh composed of right-angled triangles.
    
    Args:
        width, height: Physical dimensions of the domain.
        nx, ny: Number of intervals (cells) along X and Y.
    """
    mesh = Mesh()
    
    # 1. Create Nodes (Deterministic Grid)
    # We store IDs in a 2D list/array so we can find them easily to make cells.
    # node_grid[j][i] will hold the Node ID at row j, column i.
    node_grid = np.zeros((ny + 1, nx + 1), dtype=int)
    
    dx = width / nx
    dy = height / ny
    
    print(f"Creating Nodes ({nx+1} x {ny+1})...")
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            # Create the node in the Mesh manager
            n = mesh.add_node(x, y)
            node_grid[j, i] = n.id

    # 2. Create Cells (The "Cookie Cutter")
    # We march through the grid squares and cut each into 2 triangles.
    print(f"Creating Cells ({nx} x {ny} x 2)...")
    for j in range(ny):
        for i in range(nx):
            # Get the 4 corner Node IDs of this square
            # n3 -- n4
            # |     |
            # n1 -- n2
            id1 = node_grid[j,   i]
            id2 = node_grid[j,   i+1]
            id3 = node_grid[j+1, i]
            id4 = node_grid[j+1, i+1]
            
            # Triangle A (Bottom-Left)
            # Connectivity: (i,j) -> (i+1,j) -> (i+1, j+1)
            mesh.add_cell(id1, id2, id4)
            
            # Triangle B (Top-Right)
            # Connectivity: (i,j) -> (i+1,j+1) -> (i, j+1)
            mesh.add_cell(id1, id4, id3)
            
    return mesh

def run():
    print("--- Structured Grid Test ---")
    
    # Generate a simple 1.0 x 1.0 square with 10 divisions
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 10, 10
    
    mesh = create_structured_grid(Lx, Ly, Nx, Ny)
    
    print(f"Generated {len(mesh.nodes)} nodes and {len(mesh.cells)} cells.")
    
    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    
    # Extract points for plotting
    # (Matplotlib needs lists of coordinates)
    x_vals = []
    y_vals = []
    triangles = []
    
    # Map Node IDs to 0..N indices for matplotlib's triplot
    # (Because our Mesh IDs might start at 1 or 100)
    node_id_to_idx = {}
    for idx, node in enumerate(mesh.nodes.values()):
        x_vals.append(node.x)
        y_vals.append(node.y)
        node_id_to_idx[node.id] = idx
        
    for cell in mesh.cells.values():
        triangles.append([
            node_id_to_idx[cell.n1.id],
            node_id_to_idx[cell.n2.id],
            node_id_to_idx[cell.n3.id]
        ])
        
    ax.triplot(x_vals, y_vals, triangles, 'k-', lw=0.5)
    ax.set_title(f"Structured Grid ({Nx}x{Ny})")
    plt.show()

if __name__ == "__main__":
    run()