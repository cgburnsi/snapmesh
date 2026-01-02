"""
ex06a_supersonic_bump.py
------------------------
The Flagship Simulation.
Combines:
  1. Geometry (Channel with Circular Bump)
  2. Mesh Generation (Unstructured Triangles)
  3. FVM Solver (Euler Equations)
  4. Visualization (Matplotlib Contours)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment, Arc
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.euler import Euler2D

def create_bump_mesh():
    m = Mesh()
    
    # Domain: Length=3, Height=1
    # Bump: centered at x=1.5, height=0.2, chord=1.0
    
    # Points
    p0 = [0,0]; p1 = [1,0]          # Up to bump
    p2 = [2,0]; p3 = [3,0]          # After bump
    p4 = [3,1]; p5 = [0,1]          # Top
    
    # Bump Geometry (Arc)
    # R^2 = (R-h)^2 + (c/2)^2  => R = (h^2 + c^2/4) / 2h
    h = 0.2; c = 1.0
    R = (h**2 + c**2/4.0) / (2*h) # R = 0.725
    
    # Center is at (1.5, -R + h)
    center = [1.5, -R + h]
    
    # Angles for the Arc
    t1 = np.arctan2(0 - center[1], 1 - center[0])
    t2 = np.arctan2(0 - center[1], 2 - center[0])
    
    # --- BOUNDARIES ---
    # Tagging is essential for the Solver to know what is what
    l_inlet = LineSegment(p5, p0, name="inlet")
    l_bot1  = LineSegment(p0, p1, name="wall")
    l_bump  = Arc(center, R, t1, t2, name="wall")
    l_bot2  = LineSegment(p2, p3, name="wall")
    l_outlet= LineSegment(p3, p4, name="outlet")
    l_top   = LineSegment(p4, p5, name="wall")
    
    m.add_curve(l_inlet)
    m.add_curve(l_bot1)
    m.add_curve(l_bump)
    m.add_curve(l_bot2)
    m.add_curve(l_outlet)
    m.add_curve(l_top)
    
    # Sizing: Vectorized for DistMesh
    def sizing(x, y):
        # Calculate distance for inputs (works for scalars and arrays)
        dist_bump = np.sqrt((x - 1.5)**2 + y**2)
        return np.where(dist_bump < 1.0, 0.05, 0.15)
        
    m.discretize_boundary(sizing)
    generate_unstructured_mesh(m, sizing, h_base=0.10, n_smooth=20)
    return m

def plot_solution(mesh, grid, solver, title="Flow Field"):
    """ Visualizes Mach Number AND Pressure fields """
    print("\n--- Generating Visualization ---")
    
    # 1. Build Connectivity for Triangulation
    sorted_nodes = sorted(mesh.nodes.values(), key=lambda n: n.id)
    node_id_map = {n.id: i for i, n in enumerate(sorted_nodes)}
    
    triangles = []
    for c in mesh.cells.values():
        triangles.append([
            node_id_map[c.n1.id],
            node_id_map[c.n2.id],
            node_id_map[c.n3.id]
        ])
        
    # 2. Calculate Cell Data
    rho = solver.q[:,0]
    u = solver.q[:,1]/rho
    v = solver.q[:,2]/rho
    
    # Calculate Pressure
    p = (1.4-1)*(solver.q[:,3] - 0.5*rho*(u**2+v**2))
    p = np.maximum(p, 1e-5) # Clamp for visualization safety
    
    # Calculate Mach
    c_sound = np.sqrt(1.4 * p / rho)
    mach = np.sqrt(u**2 + v**2) / c_sound
    
    # 3. Plot Setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    triang = tri.Triangulation(grid.nodes_x, grid.nodes_y, triangles)
    
    # Plot 1: Mach Number
    img1 = ax1.tripcolor(triang, facecolors=mach, cmap='turbo', edgecolors='none')
    ax1.set_title(f"{title} - Mach Number")
    ax1.set_ylabel("Y [m]")
    ax1.set_aspect('equal')
    plt.colorbar(img1, ax=ax1, label="Mach")
    
    # Plot 2: Pressure
    img2 = ax2.tripcolor(triang, facecolors=p, cmap='inferno', edgecolors='none')
    ax2.set_title(f"{title} - Pressure [Pa]")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_aspect('equal')
    plt.colorbar(img2, ax=ax2, label="Pressure [Pa]")
    
    plt.tight_layout()
    plt.show()

def run():
    print("--- Supersonic Flow over Bump (Mach 1.5) ---")
    
    # 1. Mesh & Solver
    mesh = create_bump_mesh()
    grid = Grid(mesh)
    model = Euler2D()
    
    # Initialize as First Order for stability
    solver = FiniteVolumeSolver(grid, model, order=1)
    
    # 2. Conditions
    gamma = 1.4; rho_in = 1.0; p_in = 100000.0
    c_in = np.sqrt(gamma * p_in / rho_in)
    M_in = 1.5; u_in = M_in * c_in
    E_in = p_in/(gamma-1) + 0.5*rho_in*u_in**2
    q_inlet = np.array([rho_in, rho_in*u_in, 0.0, E_in])
    
    model.set_boundary_value("inlet", q_inlet)
    solver.set_initial_condition(np.tile(q_inlet, (grid.n_cells, 1)))
    
    # 3. Time Marching (Warm Start Strategy)
    steps = 1500
    print(f"\nStarting Solver ({steps} steps)...")
    print("   -> Phase 1: First-Order (Smoothing Transients)")
    
    for i in range(steps + 1):
        # SWITCH TO SECOND ORDER AFTER 100 STEPS
        if i == 100:
            print("   -> Phase 2: Switching to Second-Order (Sharpening Shocks)")
            solver.order = 2
            
        resid = solver.step(dt=1e-5)
        
        if i % 100 == 0:
            rho = solver.q[:,0]
            # No need for manual decode, model handles clamping now, but for print:
            p = (1.4-1)*(solver.q[:,3] - 0.5*(solver.q[:,1]**2+solver.q[:,2]**2)/np.maximum(rho, 1e-12))
            
            print(f"   Iter {i:4d}: Resid={resid:.2e} | P_range=[{np.min(p):.0f}, {np.max(p):.0f}] Pa")

    plot_solution(mesh, grid, solver, title=f"Mach 1.5 Bump (Order=2 MUSCL)")

if __name__ == "__main__":
    run()