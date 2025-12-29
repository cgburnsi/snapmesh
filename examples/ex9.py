import snapmesh as sm
import snapfvm.grid as fvm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_connectivity():
    print("Building Coarse Test Mesh...")
    
    m = sm.Mesh()
    
    # 1. Create Nodes for a simple 1x1 Square
    n1 = m.add_node(0.0, 0.0)
    n2 = m.add_node(1.0, 0.0)
    n3 = m.add_node(1.0, 1.0)
    n4 = m.add_node(0.0, 1.0)
    
    # 2. Create Cells (THIS WAS MISSING BEFORE)
    # Split the square into 2 triangles
    # Triangle 1: Bottom-Right
    m.add_cell(n1.id, n2.id, n3.id)
    # Triangle 2: Top-Left
    m.add_cell(n1.id, n3.id, n4.id)
    
    # 3. Tag Boundaries (So we get red arrows)
    m.tag_boundary_edge(n1.id, n2.id, "Bottom")
    m.tag_boundary_edge(n2.id, n3.id, "Right")
    m.tag_boundary_edge(n3.id, n4.id, "Top")
    m.tag_boundary_edge(n4.id, n1.id, "Left")
    
    print(f"Initial Mesh: {len(m.cells)} cells.")
    
    # 4. Refine once to make it interesting (2 -> 8 triangles)
    m = sm.refine_global(m) 
    print(f"Refined Mesh: {len(m.cells)} cells.")

    # 5. Convert to FVM Grid
    print("Converting to UnstructuredGrid...")
    grid = fvm.UnstructuredGrid(m)
    
    # 6. Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # A. Draw Edges (Background Mesh)
    # Helper to convert node IDs to 0..N indices for plotting
    triangles = []
    for c in m.cells.values():
        idx1 = grid.node_id_map[c.n1]
        idx2 = grid.node_id_map[c.n2]
        idx3 = grid.node_id_map[c.n3]
        triangles.append([idx1, idx2, idx3])
        
    ax.triplot(grid.nodes_x, grid.nodes_y, triangles, color='k', alpha=0.2, lw=1)
               
    # B. Draw Cell Centroids (Blue Dots)
    ax.plot(grid.cell_centers_x, grid.cell_centers_y, 'o', color='tab:blue', markersize=5)
    
    # C. Draw Connectivity Arrows
    for i in range(grid.num_faces):
        owner_idx = grid.face_owner[i]
        neigh_idx = grid.face_neighbor[i]
        
        # Start Point: Owner Centroid
        start_x = grid.cell_centers_x[owner_idx]
        start_y = grid.cell_centers_y[owner_idx]
        
        if neigh_idx != -1:
            # INTERNAL FACE (Green): Owner -> Neighbor
            end_x = grid.cell_centers_x[neigh_idx]
            end_y = grid.cell_centers_y[neigh_idx]
            
            # Draw arrow (Green)
            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", color='tab:green', lw=2, shrinkA=5, shrinkB=5))
        else:
            # BOUNDARY FACE (Red): Owner -> Face Center
            end_x = grid.face_centers_x[i]
            end_y = grid.face_centers_y[i]
            
            # Draw arrow (Red)
            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", color='tab:red', lw=2, shrinkA=5, shrinkB=0))

    ax.set_aspect('equal')
    ax.set_title(f"FVM Connectivity Check\nGreen: Internal Flux | Red: Boundary Flux")
    
    # Legend
    custom_lines = [Line2D([0], [0], color='tab:green', lw=2),
                    Line2D([0], [0], color='tab:red', lw=2),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=8)]
    ax.legend(custom_lines, ['Internal (Owner->Neighbor)', 'Boundary (Owner->Face)', 'Cell Centroid'])
    
    plt.show()

if __name__ == '__main__':
    visualize_connectivity()