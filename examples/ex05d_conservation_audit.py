"""
ex05d_conservation_audit.py
---------------------------
Goal: Run a short simulation to verify the Solver's Conservation Audit.
"""
import numpy as np
from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.euler import Euler2D

def create_box_mesh():
    m = Mesh()
    # 1x1 Box
    l1 = LineSegment([0,0], [1,0]); m.add_curve(l1)
    l2 = LineSegment([1,0], [1,1]); m.add_curve(l2)
    l3 = LineSegment([1,1], [0,1]); m.add_curve(l3)
    l4 = LineSegment([0,1], [0,0]); m.add_curve(l4)
    m.discretize_boundary(lambda x,y: 0.1)
    generate_unstructured_mesh(m, lambda x,y: 0.1, h_base=0.1, n_smooth=10)
    return m

def run():
    print("--- Conservation Audit Test ---")
    
    # 1. Setup
    mesh = create_box_mesh()
    grid = Grid(mesh)
    model = Euler2D()
    solver = FiniteVolumeSolver(grid, model)
    
    # 2. Initial Condition (Explosion)
    print("   -> Setting Initial Condition (High Pressure Center)...")
    rho = np.ones(grid.n_cells)
    u = np.zeros(grid.n_cells)
    v = np.zeros(grid.n_cells)
    p = np.ones(grid.n_cells) * 100000.0
    
    # High pressure zone in middle
    cx, cy = grid.cell_centers[:,0], grid.cell_centers[:,1]
    mask = (cx - 0.5)**2 + (cy - 0.5)**2 < 0.2**2
    p[mask] = 500000.0
    
    q_init = np.zeros((grid.n_cells, 4))
    q_init[:,0] = rho
    q_init[:,1] = rho*u
    q_init[:,2] = rho*v
    q_init[:,3] = p / (1.4 - 1.0) # Internal Energy (u=0)
    
    solver.set_initial_condition(q_init)
    
    # 3. Run
    print("\nStarting Time Marching (50 steps)...")
    initial_mass = solver.total_mass_history[0]
    
    for i in range(50):
        solver.step(dt=1e-5)
        
    final_mass = solver.total_mass_history[-1]
    diff = final_mass - initial_mass
    
    print(f"\nInitial Mass: {initial_mass:.8f}")
    print(f"Final Mass:   {final_mass:.8f}")
    print(f"Discrepancy:  {diff:.2e}")
    
    if abs(diff) < 1e-12:
        print("SUCCESS: Global Conservation Verified.")
    else:
        print("WARNING: Mass leakage detected.")

if __name__ == "__main__":
    run()