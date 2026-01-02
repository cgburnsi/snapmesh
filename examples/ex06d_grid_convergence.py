
"""
ex06d_grid_convergence.py
-------------------------
The Final Verdict.
Runs the Lid-Driven Cavity at 3 different resolutions to prove
that the error vanishes as the mesh gets finer (Consistency).
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

# Ghia Data
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
                   0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
GHIA_U = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, 
                   -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

def run_case(h_size, steps):
    print(f"\n--- Running Case: h={h_size} ---")
    
    # 1. Generate Mesh
    m = Mesh()
    p0=[0,0]; p1=[1,0]; p2=[1,1]; p3=[0,1]
    m.add_curve(LineSegment(p0, p1, name="wall"))        
    m.add_curve(LineSegment(p1, p2, name="wall"))        
    m.add_curve(LineSegment(p2, p3, name="moving_wall")) 
    m.add_curve(LineSegment(p3, p0, name="wall"))        
    
    def sizing(x, y): return np.full_like(x, h_size)
    m.discretize_boundary(sizing)
    generate_unstructured_mesh(m, sizing, h_base=h_size, n_smooth=15)
    
    grid = Grid(m)
    print(f"   -> Grid: {grid.n_cells} Cells")
    
    # 2. Solver Setup (2nd Order)
    model = IncompressibleAC(mu=0.01, beta=1.5)
    model.set_boundary_value("moving_wall", [0, 1.0, 0.0])
    
    solver = FiniteVolumeSolver(grid, model, order=2)
    solver.set_initial_condition(np.zeros((grid.n_cells, 3)))
    
    # 3. Run
    # Time step scales with h to maintain CFL stability
    dt = 0.001 * (h_size / 0.04) 
    
    # Standard loop range since we don't print inside
    for i in range(steps):
        solver.step(dt)
        
        # Status Update every 500 steps
        if i % 10 == 0:
             print(f"   Iter {i:4d}: ")

        
    # 4. Extract Profile
    triang = tri.Triangulation(grid.cell_centers[:,0], grid.cell_centers[:,1])
    interp_u = tri.LinearTriInterpolator(triang, solver.q[:, 1])
    
    u_profile = interp_u(np.full_like(GHIA_Y, 0.5), GHIA_Y)
    u_profile[0] = 0.0; u_profile[-1] = 1.0 # Enforce BC
    
    return u_profile, grid.n_cells

def run():
    print("--- Grid Convergence Study ---")
    
    # Define Cases
    cases = [
        {"h": 0.06,  "steps": 2000, "label": "Coarse"},
        {"h": 0.03,  "steps": 4000, "label": "Medium"},
        {"h": 0.015, "steps": 8000, "label": "Fine"}
    ]
    
    plt.figure(figsize=(8, 7))
    plt.plot(GHIA_U, GHIA_Y, 'ko', label='Benchmark (Ghia 1982)', markersize=6, zorder=10)
    
    colors = ['g--', 'b-', 'r-']
    
    for i, case in enumerate(cases):
        u_prof, n_cells = run_case(case["h"], case["steps"])
        
        # Calculate Error
        err = np.sqrt(np.mean((u_prof - GHIA_U)**2))
        
        label_str = f"{case['label']} ({n_cells} cells) - Err: {err:.4f}"
        plt.plot(u_prof, GHIA_Y, colors[i], label=label_str, linewidth=2)
        
    plt.xlabel("U-Velocity")
    plt.ylabel("Y-Coordinate")
    plt.title("Grid Convergence Study (Re=100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()