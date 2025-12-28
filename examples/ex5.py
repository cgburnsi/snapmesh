import snapmesh as sm
import matplotlib.pyplot as plt

def plot_mesh(m, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    for e in m.edges.values():
        n1 = m.nodes[e.n1]
        n2 = m.nodes[e.n2]
        # Color boundary edges red, internal edges blue
        color = 'r-' if e.is_boundary else 'b-'
        lw = 1.5 if e.is_boundary else 0.5
        ax.plot([n1.x, n2.x], [n1.y, n2.y], color, lw=lw)
    ax.set_aspect('equal')
    ax.set_title(f"{title} ({len(m.cells)} cells)")
    plt.show()

if __name__ == '__main__':
    # 1. Create Coarse Nozzle
    # (Same geometry as before)
    wall_curve = sm.CubicCurve(0, 10, a=0.008, b=-0.12, c=0.2, d=3.0)
    inlet = sm.Line(0, 0, 0, 3)
    outlet = sm.Line(10, 2, 10, 0)
    centerline = sm.Line(10, 0, 0, 0)
    nozzle = sm.CompositeCurve([inlet, wall_curve, outlet, centerline])

    m = sm.Mesh()
    # Start VERY coarse (only 4 segments per side) to prove the refinement works
    m.add_boundary_loop(nozzle, count=40, tag="Wall")
    
    # Manually fill interior with a few coarse triangles for demo
    # (In a real app, you'd use an algorithm, but let's just make one manually here)
    # Actually, let's just leave it as a loop for now to see the boundary refinement.
    # Refinement works on Edges, so even a hollow loop refines perfectly.
    
    plot_mesh(m, "Level 0: Coarse")

    # 2. Refine Level 1
    m1 = sm.refine_global(m)
    plot_mesh(m1, "Level 1: Refined")

    # 3. Refine Level 2
    m2 = sm.refine_global(m1)
    plot_mesh(m2, "Level 2: Smoother")