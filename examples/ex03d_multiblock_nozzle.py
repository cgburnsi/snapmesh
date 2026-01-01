"""
ex03d_multiblock_nozzle.py
--------------------------
Goal: Generate a full De Laval Nozzle by stitching 3 mapped blocks.
Demonstrates: Multi-block generation and Node Merging.
"""
import numpy as np
import matplotlib.pyplot as plt
from snapmesh.mesh import Mesh
from snapmesh.quality import MeshQuality

# Re-use our mapping function (The "Transfinite Interpolator")
def transfinite_map(u, v, c1, c2, c3, c4):
    phi1 = (1-u)*(1-v); phi2 = u*(1-v); phi3 = u*v; phi4 = (1-u)*v
    x = phi1*c1[0] + phi2*c2[0] + phi3*c3[0] + phi4*c4[0]
    y = phi1*c1[1] + phi2*c2[1] + phi3*c3[1] + phi4*c4[1]
    return x, y

def add_mapped_block(mesh, c1, c2, c3, c4, nx, ny):
    """
    Adds a mapped grid to an EXISTING mesh object.
    """
    node_ids = np.zeros((ny + 1, nx + 1), dtype=int)
    
    # 1. Create Nodes
    for j in range(ny + 1):
        for i in range(nx + 1):
            u, v = i / nx, j / ny
            px, py = transfinite_map(u, v, c1, c2, c3, c4)
            n = mesh.add_node(px, py)
            node_ids[j, i] = n.id

    # 2. Create Cells
    for j in range(ny):
        for i in range(nx):
            id1, id2 = node_ids[j, i], node_ids[j, i+1]
            id3, id4 = node_ids[j+1, i], node_ids[j+1, i+1]
            mesh.add_cell(id1, id2, id4)
            mesh.add_cell(id1, id4, id3)

def run():
    print("--- Multi-Block Nozzle Generator ---")
    mesh = Mesh()
    
    # --- Define Geometry Points (De Laval Nozzle) ---
    # We define 3 blocks. Notice the shared coordinates at the interfaces.
    
    # Block 1: Inlet (Straight Cylinder)
    # x: 0.0 -> 1.0, Radius: 1.0
    p1_bl, p1_br = (0.0, 0.0), (1.0, 0.0)
    p1_tr, p1_tl = (1.0, 1.0), (0.0, 1.0)
    
    # Block 2: Throat (Converging)
    # x: 1.0 -> 1.5, Radius: 1.0 -> 0.4
    p2_bl, p2_br = (1.0, 0.0), (1.5, 0.0)
    p2_tr, p2_tl = (1.5, 0.4), (1.0, 1.0)
    
    # Block 3: Expansion (Diverging)
    # x: 1.5 -> 3.0, Radius: 0.4 -> 0.8
    p3_bl, p3_br = (1.5, 0.0), (3.0, 0.0)
    p3_tr, p3_tl = (3.0, 0.8), (1.5, 0.4)
    
    # --- Generate Blocks ---
    print("1. Generating Inlet Block...")
    add_mapped_block(mesh, p1_bl, p1_br, p1_tr, p1_tl, nx=10, ny=10)
    
    print("2. Generating Throat Block...")
    add_mapped_block(mesh, p2_bl, p2_br, p2_tr, p2_tl, nx=5, ny=10)
    
    print("3. Generating Expansion Block...")
    add_mapped_block(mesh, p3_bl, p3_br, p3_tr, p3_tl, nx=15, ny=10)
    
    print(f"   -> Raw Mesh: {len(mesh.nodes)} nodes")
    print("   (At this stage, the mesh has cracks at x=1.0 and x=1.5)")
    
    # --- Merge Seams ---
    print("4. Merging Interfaces...")
    n_removed = mesh.merge_duplicate_nodes()
    print(f"   -> Merged {n_removed} duplicate nodes.")
    print(f"   -> Final Mesh: {len(mesh.nodes)} nodes.")
    
    if n_removed == 0:
        print("   [!] WARNING: No nodes were merged. Check your coordinates!")
    
    # --- Validate ---
    inspector = MeshQuality(mesh)
    inspector.print_report()
    
    # --- Visualize ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect('equal')
    
    x_vals, y_vals, tris = [], [], []
    id_map = {n.id: i for i, n in enumerate(mesh.nodes.values())}
    
    for n in mesh.nodes.values():
        x_vals.append(n.x); y_vals.append(n.y)
    for c in mesh.cells.values():
        tris.append([id_map[c.n1.id], id_map[c.n2.id], id_map[c.n3.id]])
        
    ax.triplot(x_vals, y_vals, tris, 'k-', lw=0.5)
    
    # Draw interface lines to show where the stitches happened
    plt.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='Interface 1')
    plt.axvline(1.5, color='r', linestyle='--', alpha=0.5, label='Interface 2')
    
    ax.set_title("Full Nozzle Mesh (Multi-Block Stitched)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()