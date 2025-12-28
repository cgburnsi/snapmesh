import snapmesh as sm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Building Stadium Mesh...")

    # 1. Define the 4 segments of a Stadium
    #    Bottom Line: (-5, -5) to (5, -5)
    l_bot = sm.Line(-5, -5, 5, -5)

    #    Right Arc: Center at (5, 0), Radius 5, going from -90 deg (bottom) to +90 deg (top)
    a_right = sm.Arc(5, 0, 5.0, -90, 90)

    #    Top Line: (5, 5) to (-5, 5)
    l_top = sm.Line(5, 5, -5, 5)

    #    Left Arc: Center at (-5, 0), Radius 5, going from 90 deg (top) to 270 deg (bottom)
    a_left = sm.Arc(-5, 0, 5.0, 90, 270)

    # 2. Combine them
    #    Note: Order matters! They must connect head-to-tail.
    stadium = sm.CompositeCurve([l_bot, a_right, l_top, a_left])

    # 3. Generate Mesh
    m = sm.Mesh()
    
    #    'count=40' means 40 nodes total (10 per segment)
    m.add_boundary_loop(stadium, count=40, tag="TrackWall")

    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot edges
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        ax.plot([n1.x, n2.x], [n1.y, n2.y], 'b-o', markersize=4)

    # Make it look right
    ax.set_aspect('equal')
    ax.set_title("Composite Geometry: Stadium")
    plt.grid(True, alpha=0.3)
    plt.show()