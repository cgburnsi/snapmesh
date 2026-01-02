"""
ex06c_cavity_validation_tuned.py
--------------------------------
High-Fidelity Validation of Lid-Driven Cavity (Re=100).
Changes:
  1. Uniform Fine Mesh (Resolves the Core Vortex).
  2. Lower Beta (Reduces Numerical Smearing).
  3. Longer Run Time (Ensures full convergence).
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

# Ghia (1982) Benchmark Data
GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 
          0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
GHIA_U = [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, 
          -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000]

def create_uniform_fine_mesh():
    m = Mesh()
    p0=[0,0]; p1=[1,0]; p2=[1,1]; p3=[0,1]
    m.add_curve(LineSegment(p0, p1, name="wall"))        
    m.add_curve(LineSegment(p1, p2, name="wall"))        
    m.add_curve(LineSegment(p2, p3, name="moving_wall")) 
    m.add_curve(LineSegment(p3, p0, name="wall"))        
    
    # Uniform h=0.02 (approx 5000 cells)
    # This resolves the core vortex AND the walls without compromise.
    h = 0.02
    def sizing(x, y): return np.full_like(x, h)
    
    m.discretize_boundary(sizing)
    generate_unstructured_mesh(m, sizing, h_base=h, n_smooth=20)
    return m

def run():
    print("--- Tuned Validation: Re=100 (Uniform Fine Mesh) ---")
    
    # 1. Mesh
    mesh = create_uniform_fine_mesh()
    grid = Grid(mesh)
    print(f"   -> Grid: {grid.n_cells} Cells")
    
    # 2. Solver Setup
    # Lower Beta (1.0) -> Less Artificial Viscosity -> Sharper Results
    model = IncompressibleAC(mu=0.01, beta=1.0)
    model.set_boundary_value("moving_wall", [0, 1.0, 0.0])
    
    # Second Order Accuracy
    solver = FiniteVolumeSolver(grid, model, order=2)
    solver.set_initial_condition(np.zeros((grid.n_cells, 3)))
    
    # 3. Run
    # With smaller h and smaller beta, we need ample steps.
    # dt limit approx h / (U + beta) = 0.02 / 2.0 = 0.01
    # We use 0.001 to be safe and accurate.
    dt = 0.001
    steps = 15000 # 15 seconds physical time (enough for ~3 turnovers)
    
    print(f"   -> Running Solver ({steps} steps)...")
    for i in range(steps + 1):
        resid = solver.step(dt)
        if i % 1000 == 0:
             print(f"   Iter {i:5d}: Resid={resid:.2e}")

    # 4. Analysis
    print("\n   -> Extracting Profile...")
    triang = tri.Triangulation(grid.cell_centers[:,0], grid.cell_centers[:,1])
    interp_u = tri.LinearTriInterpolator(triang, solver.q[:, 1])
    
    y_query = np.array(GHIA_Y)
    u_sim = interp_u(np.full_like(y_query, 0.5), y_query)
    u_sim[0] = 0.0; u_sim[-1] = 1.0
    
    error_L2 = np.sqrt(np.mean((u_sim - GHIA_U)**2))
    print(f"   L2 Error Norm: {error_L2:.5f}")
    
    # 5. Plot
    plt.figure(figsize=(7, 6))
    plt.plot(GHIA_U, GHIA_Y, 'ko', label='Ghia et al. (1982)', markersize=6)
    plt.plot(u_sim, GHIA_Y, 'r-', label=f'SnapFVM (Tuned)', linewidth=2)
    
    plt.xlabel("U-Velocity")
    plt.ylabel("Y-Coordinate")
    plt.title(f"Tuned Validation (Re=100)\nCells: {grid.n_cells}, Error: {error_L2:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()