"""
ex06c_cavity_validation.py
--------------------------
Quantitative Validation of Lid-Driven Cavity (Re=100).
UPDATED: Uses Spatially Adaptive Meshing (Fine Walls, Coarse Center).
         Achieves higher accuracy with fewer cells.
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

def create_adaptive_mesh():
    m = Mesh()
    p0=[0,0]; p1=[1,0]; p2=[1,1]; p3=[0,1]
    m.add_curve(LineSegment(p0, p1, name="wall"))        
    m.add_curve(LineSegment(p1, p2, name="wall"))        
    m.add_curve(LineSegment(p2, p3, name="moving_wall")) 
    m.add_curve(LineSegment(p3, p0, name="wall"))        
    
    # --- INTELLIGENT SIZING ---
    # Wall h = 0.01 (Very Fine)
    # Bulk h = 0.06 (Coarse)
    # This puts cells only where the physics happens.
    def sizing(x, y):
        # Distance to nearest wall
        d_x = np.minimum(x, 1.0 - x)
        d_y = np.minimum(y, 1.0 - y)
        dist = np.minimum(d_x, d_y)
        
        h_min = 0.01
        h_max = 0.06
        
        # Ramp from h_min to h_max over 0.15 distance
        h = h_min + (h_max - h_min) * (dist / 0.15)
        return np.minimum(h, h_max)
    
    m.discretize_boundary(sizing)
    # Use fewer smoothing iterations to preserve the gradient density
    generate_unstructured_mesh(m, sizing, h_base=0.06, n_smooth=15)
    return m

def run():
    print("--- Quantitative Validation: Adaptive Mesh (Re=100) ---")
    
    # 1. Generate Efficient Mesh
    mesh = create_adaptive_mesh()
    grid = Grid(mesh)
    print(f"   -> Generated Adaptive Grid: {grid.n_cells} Cells")
    
    # 2. Solver Setup
    model = IncompressibleAC(mu=0.01, beta=2.0)
    model.set_boundary_value("moving_wall", [0, 1.0, 0.0])
    
    # Use Second Order for best accuracy per cell
    solver = FiniteVolumeSolver(grid, model, order=2)
    solver.set_initial_condition(np.zeros((grid.n_cells, 3)))
    
    # 3. Run
    # Since small cells limit dt, we need a conservative time step
    dt = 0.0004 
    steps = 6000 # More steps needed for smaller dt
    
    print(f"   -> Running Solver ({steps} steps)...")
    for i in range(steps + 1):
        resid = solver.step(dt)
        if i % 1000 == 0:
             print(f"   Iter {i:4d}: Resid={resid:.2e}")

    # 4. Analysis
    print("\n   -> Extracting Profile...")
    triang = tri.Triangulation(grid.cell_centers[:,0], grid.cell_centers[:,1])
    interp_u = tri.LinearTriInterpolator(triang, solver.q[:, 1])
    
    u_sim = interp_u(np.full_like(GHIA_Y, 0.5), GHIA_Y)
    u_sim[0] = 0.0; u_sim[-1] = 1.0
    
    error_L2 = np.sqrt(np.mean((u_sim - GHIA_U)**2))
    print(f"   L2 Error Norm: {error_L2:.5f}")
    
    # 5. Plot
    plt.figure(figsize=(7, 6))
    
    # Plot Mesh Density in background to show the adaptation
    plt.tripcolor(triang, solver.q[:,1], cmap='Greys', alpha=0.3)
    
    plt.plot(GHIA_U, GHIA_Y, 'ko', label='Ghia et al. (1982)')
    plt.plot(u_sim, GHIA_Y, 'r-', label=f'Adaptive SnapFVM', linewidth=2)
    
    plt.xlabel("U-Velocity")
    plt.ylabel("Y-Coordinate")
    plt.title(f"Adaptive Mesh Validation (Re=100)\nCells: {grid.n_cells}, Error: {error_L2:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()