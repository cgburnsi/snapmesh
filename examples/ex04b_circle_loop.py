"""
ex04b_circle_loop.py
--------------------
Goal: Use the Mesh Manager to generate a boundary loop automatically.
"""
import matplotlib.pyplot as plt
from snapmesh.mesh import Mesh, BCTag

def run():
    print("--- 1. Setup Mesh ---")
    mesh = Mesh()
    
    # 1. Create and Register Geometry
    # We use the factory method to keep the mesh manager in charge.
    gid = mesh.add_circle((0,0), 2.0)
    
    print(f"Registered Circle (ID {gid})")

    print("\n--- 2. Generate Loop ---")
    # Generate 16 nodes along the circle
    # Note: We use 'bc_tag' here to match the definition in mesh.py
    nodes = mesh.add_boundary_loop(gid, 16, bc_tag=BCTag.WALL)
    
    print(f"Generated {len(nodes)} nodes and {len(mesh.edges)} edges.")

    print("\n--- 3. Visualize ---")
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # Plot Edges
    for edge in mesh.edges.values():
        p1 = edge.node_a.to_array()
        p2 = edge.node_b.to_array()
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', marker='o')
        
    plt.title("Generated Boundary Loop")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run()