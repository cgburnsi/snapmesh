import snapmesh as sm
import snapfvm.grid as fvm
import matplotlib.pyplot as plt
import numpy as np

def check_normals():
    print("Building Test Mesh...")
    m = sm.Mesh()
    
    # Simple 2-triangle square again
    n1 = m.add_node(0.0, 0.0)
    n2 = m.add_node(1.0, 0.0)
    n3 = m.add_node(1.0, 1.0)
    n4 = m.add_node(0.0, 1.0)
    m.add_cell(n1.id, n2.id, n3.id)
    m.add_cell(n1.id, n3.id, n4.id)
    m.tag_boundary_edge(n1.id, n2.id, "Bottom")
    m.tag_boundary_edge(n2.id, n3.id, "Right")
    m.tag_boundary_edge(n3.id, n4.id, "Top")
    m.tag_boundary_edge(n4.id, n1.id, "Left")
    
    # Refine once
    m = sm.refine_global(m) 
    
    # Convert
    grid = fvm.UnstructuredGrid(m)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Background Mesh
    triangles = []
    for c in m.cells.values():
        triangles.append([grid.node_id_map[c.n1], 
                          grid.node_id_map[c.n2], 
                          grid.node_id_map[c.n3]])
    ax.triplot(grid.nodes_x, grid.nodes_y, triangles, 'k-', alpha=0.2)
    
    # 2. Cell Centers (Blue Dots)
    ax.plot(grid.cell_centers_x, grid.cell_centers_y, 'bo', markersize=4, label="Owner Cell")

    # 3. Normals (Red Arrows)
    # We draw them starting at the Face Center
    quiver = ax.quiver(grid.face_centers_x, grid.face_centers_y,
                       grid.face_normals_x, grid.face_normals_y,
                       color='r', scale=15, width=0.005, label="Face Normal")

    ax.set_aspect('equal')
    ax.set_title("Face Normals Check (Must point OUT of Blue Dots)")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    check_normals()