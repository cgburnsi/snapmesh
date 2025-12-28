import snapmesh as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 1. Define the Math
    geom = sm.Circle(0, 0, 10.0)
    
    # 2. Build the Discretization
    m = sm.Mesh()
    m.add_boundary_loop(geom, count=50, tag="Wall")
    
    # 3. Simple Plot
    print(f"Created mesh with {len(m.nodes)} nodes.")
    
    fig, ax = plt.subplots()
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        ax.plot([n1.x, n2.x], [n1.y, n2.y], 'b-')
    
    ax.set_aspect('equal')
    plt.show()