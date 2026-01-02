"""
ex06b_lid_driven_cavity.py
--------------------------
Incompressible Flow (Re=100).
Top wall moves to the right.
Solves [p, u, v] using Artificial Compressibility.
FIXED: Visualization now correctly maps Cell data to Nodes for Streamlines.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.incompressible import IncompressibleAC

def create_cavity_mesh():
    m = Mesh()
    # 1x1 Square
    p0=[0,0]; p1=[1,0]; p2=[1,1]; p3=[0,1]
    
    # Tagging is critical
    m.add_curve(LineSegment(p0, p1, name="wall"))        # Bottom
    m.add_curve(LineSegment(p1, p2, name="wall"))        # Right
    m.add_curve(LineSegment(p2, p3, name="moving_wall")) # Top (LID)
    m.add_curve(LineSegment(p3, p0, name="wall"))        # Left
    
    # Uniform mesh size
    h = 0.04 # Results in ~25x25 grid
    
    def sizing(x, y):
        return np.full_like(x, h)
        
    m.discretize_boundary(sizing)
    generate_unstructured_mesh(m, sizing, h_base=h, n_smooth=20)
    return m

def run():
    print("--- Lid Driven Cavity (Re=100) ---")
    
    # 1. Setup
    mesh = create_cavity_mesh()
    grid = Grid(mesh)
    
    # Physics: Re = rho*U*L / mu = 1*1*1 / 0.01 = 100
    model = IncompressibleAC(mu=0.01, beta=1.5)
    model.set_boundary_value("moving_wall", [0, 1.0, 0.0]) # [p, u, v]
    
    # Solver
    solver = FiniteVolumeSolver(grid, model, order=1)
    solver.set_initial_condition(np.zeros((grid.n_cells, 3)))
    
    # 2. Run
    steps = 2500
    dt = 0.001 
    
    print(f"\nStarting Solver ({steps} steps)...")
    for i in range(steps + 1):
        resid = solver.step(dt)
        if i % 500 == 0:
            cx, cy = grid.cell_centers[:,0], grid.cell_centers[:,1]
            dist = (cx-0.5)**2 + (cy-0.5)**2
            idx = np.argmin(dist)
            u_center = solver.q[idx, 1]
            print(f"   Iter {i:4d}: Resid={resid:.2e} | U_center={u_center:.3f}")

    # 3. Visualization
    print("\n--- Plotting ---")
    
    # A. Build Connectivity & Node Map
    sorted_nodes = sorted(mesh.nodes.values(), key=lambda n: n.id)
    node_map = {n.id: i for i, n in enumerate(sorted_nodes)}
    
    # We must iterate cells in the SAME order as the Grid compiler did (by ID)
    sorted_cells = sorted(mesh.cells.values(), key=lambda c: c.id)
    
    triangles = []
    for c in sorted_cells:
        triangles.append([node_map[c.n1.id], node_map[c.n2.id], node_map[c.n3.id]])
        
    # B. Project Cell Data -> Nodes (Averaging)
    # This is required because Streamplots need a continuous grid, 
    # and Matplotlib interpolates from Nodes, not Cells.
    node_u = np.zeros(grid.n_nodes)
    node_v = np.zeros(grid.n_nodes)
    node_p = np.zeros(grid.n_nodes)
    node_counts = np.zeros(grid.n_nodes)
    
    for i, c in enumerate(sorted_cells):
        # Accumulate cell value to its 3 nodes
        # i is the cell index matching solver.q[i]
        nodes = [node_map[c.n1.id], node_map[c.n2.id], node_map[c.n3.id]]
        
        val_u = solver.q[i, 1]
        val_v = solver.q[i, 2]
        val_p = solver.q[i, 0]
        
        for n_idx in nodes:
            node_u[n_idx] += val_u
            node_v[n_idx] += val_v
            node_p[n_idx] += val_p
            node_counts[n_idx] += 1
            
    # Normalize
    node_u /= np.maximum(node_counts, 1)
    node_v /= np.maximum(node_counts, 1)
    node_p /= np.maximum(node_counts, 1)

    # C. Interpolate to Grid for Streamlines
    x_lin = np.linspace(0, 1, 100)
    y_lin = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    
    triang = tri.Triangulation(grid.nodes_x, grid.nodes_y, triangles)
    
    interp_u = tri.LinearTriInterpolator(triang, node_u)
    interp_v = tri.LinearTriInterpolator(triang, node_v)
    interp_p = tri.LinearTriInterpolator(triang, node_p)
    
    U_grid = interp_u(X, Y)
    V_grid = interp_v(X, Y)
    P_grid = interp_p(X, Y)
    
    # D. Plot
    plt.figure(figsize=(9, 7))
    
    # Pressure Contour
    plt.contourf(X, Y, P_grid, levels=20, cmap="viridis", alpha=0.6)
    plt.colorbar(label="Pressure (AC)")
    
    # Streamlines
    plt.streamplot(X, Y, U_grid, V_grid, color='k', linewidth=0.8, density=1.5, arrowsize=1.0)
    
    plt.title(f"Lid Driven Cavity (Re=100) - {grid.n_cells} Cells")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()