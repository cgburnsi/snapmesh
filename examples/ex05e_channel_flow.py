"""
ex05e_channel_flow.py
---------------------
Goal: Test Dirichlet (Inlet) and Neumann (Outlet) Boundary Conditions.
      Simulates Mach 2 flow flushing through a channel.
      UPDATED: Increased step count to allow flow to fully establish.
"""
import numpy as np
from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.euler import Euler2D

def create_channel_mesh():
    m = Mesh()
    # 2.0 long x 0.5 high
    # Tagging is critical here!
    l1 = LineSegment([0,0], [2,0], name="wall")
    l2 = LineSegment([2,0], [2,0.5], name="outlet") # Right
    l3 = LineSegment([2,0.5], [0,0.5], name="wall")
    l4 = LineSegment([0,0.5], [0,0], name="inlet") # Left
    
    m.add_curve(l1); m.add_curve(l2); m.add_curve(l3); m.add_curve(l4)
    
    sizing = lambda x,y: 0.1
    m.discretize_boundary(sizing)
    generate_unstructured_mesh(m, sizing, h_base=0.1, n_smooth=10)
    return m

def run():
    print("--- Channel Flow Test (Inlet -> Outlet) ---")
    
    # 1. Setup
    mesh = create_channel_mesh()
    grid = Grid(mesh)
    model = Euler2D()
    solver = FiniteVolumeSolver(grid, model)
    
    # 2. Define Inlet State (Dirichlet)
    # Mach 2.0: rho=1.0, u=680, v=0, p=100000
    gamma = 1.4
    rho_in = 1.0
    p_in = 100000.0
    c_in = np.sqrt(gamma * p_in / rho_in) # ~374 m/s
    u_in = 2.0 * c_in # Mach 2
    
    E_in = p_in/(gamma-1) + 0.5*rho_in*u_in**2
    
    q_inlet = np.array([rho_in, rho_in*u_in, 0.0, E_in])
    
    # Register this with the physics model
    model.set_boundary_value("inlet", q_inlet)
    print(f"   -> Inlet Condition Set: Mach 2.0 (u={u_in:.1f} m/s)")
    
    # 3. Initial Condition (Stagnant Air)
    # Same P and Rho, but u=0
    q_init = np.zeros((grid.n_cells, 4))
    q_init[:, 0] = rho_in
    q_init[:, 3] = p_in / (gamma - 1)
    solver.set_initial_condition(q_init)
    
    # 4. Run until flow passes through
    # Transit time = 2.0m / 748m/s = 0.0027s
    # We want 3x transit time = ~0.008s
    # dt = 1e-5 -> 800 steps minimum.
    
    print("\nStarting Time Marching (Target: 1000 steps or Convergence)...")
    
    for i in range(1001):
        resid = solver.step(dt=1e-5)
        
        if i % 100 == 0:
            rho = solver.q[:,0]
            u = solver.q[:,1] / rho
            max_u = np.max(u)
            print(f"   Iter {i}: Max Velocity = {max_u:.1f} m/s (Target: {u_in:.1f})")
            
            # Convergence Check: If we are within 0.1% of target speed
            if abs(max_u - u_in) < 1.0:
                print(f"   -> Converged early at step {i}")
                break

    # 5. Verification
    final_max_u = np.max(solver.q[:,1] / solver.q[:,0])
    print(f"\nFinal Max Velocity: {final_max_u:.1f} m/s")
    
    # Check if we are within 1% of target
    if abs(final_max_u - u_in) < 0.01 * u_in:
        print(f"SUCCESS: Flow established. Error: {abs(final_max_u - u_in):.2f} m/s")
    else:
        print(f"FAILURE: Did not reach target velocity. (Diff: {abs(final_max_u - u_in):.2f})")

if __name__ == "__main__":
    run()