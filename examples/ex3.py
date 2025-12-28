import snapmesh as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Building Convex Shapes...")

    # 1. Arbitrary Convex Shape (Triangle/Wedge)
    #    Just pass the points!
    wedge = sm.Polygon([(0, 0), (10, 0), (10, 5)])

    # 2. Regular Convex Shape (Octagon / Stop Sign)
    #    Center at (-15, 0), Radius 5, 8 sides
    stop_sign = sm.RegularPolygon(-15, 0, 5.0, 8)

    # 3. Mesh them
    m = sm.Mesh()
    
    # Add the wedge (Triangle)
    m.add_boundary_loop(wedge, count=30, tag="Wedge")
    
    # Add the stop sign
    m.add_boundary_loop(stop_sign, count=40, tag="StopSign")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        ax.plot([n1.x, n2.x], [n1.y, n2.y], 'b-o', markersize=4)

    ax.set_aspect('equal')
    ax.set_title("Convex Shapes: Polygon & RegularPolygon")
    plt.grid(True, alpha=0.3)
    plt.show()