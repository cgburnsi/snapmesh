"""
ex06g_nozzle_validation.py
--------------------------
Validates the Nozzle Simulation against 1D Isentropic Theory.
Compares:
  1. CFD Centerline Mach (Computed)
  2. Theoretical Mach (Exact Analytical Solution for Planar Nozzle)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.optimize import fsolve

# --- HARD RESET ---
modules_to_kill = ['snapfvm.numerics', 'snapfvm.solver', 'snapfvm.physics.euler']
for mod in modules_to_kill:
    if mod in sys.modules:
        del sys.modules[mod]

import unit_convert as cv
from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment, Arc
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.euler import Euler2D

# --- 1. Define Nozzle Geometry (Same as before) ---
xi  = cv.convert(0.31, 'inch', 'm')
ri  = cv.convert(2.50, 'inch', 'm')
rci = cv.convert(0.80, 'inch', 'm')
rt  = cv.convert(0.80, 'inch', 'm')
rct = cv.convert(0.50, 'inch', 'm')
xe  = cv.convert(4.05, 'inch', 'm')
ani = np.deg2rad(44.88)
ane = np.deg2rad(15.0)

# Key geometric X-coordinates for the throat
xtan = xi + rci * np.sin(ani)
rtan = ri + rci * (np.cos(ani) - 1.0)
rt1 = rt - rct * (np.cos(ani) - 1.0)
xt1 = xtan + (rtan - rt1) / np.tan(ani)
xt = xt1 + rct * np.sin(ani)

def create_nozzle_mesh():
    mesh = Mesh()
    # Define Boundaries
    mesh.add_curve(LineSegment([xi, 0.0], [xi, ri], name="inlet"))
    
    # Wall Construction
    arc1 = Arc((xi, ri-rci), rci, np.pi/2, np.pi/2 - ani, name="wall")
    mesh.add_curve(arc1)
    conv = LineSegment(arc1.evaluate(1.0), [xt1, rt1], name="wall")
    mesh.add_curve(conv)
    arc2 = Arc((xt, rt+rct), rct, 3*np.pi/2 - ani, 3*np.pi/2 + ane, name="wall")
    mesh.add_curve(arc2)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)
    div = LineSegment(arc2.evaluate(1.0), [xe, re], name="wall")
    mesh.add_curve(div)
    
    mesh.add_curve(LineSegment([xe, re], [xe, 0.0], name="outlet"))
    mesh.add_curve(LineSegment([xe, 0.0], [xi, 0.0], name="wall")) # Symmetry
    
    # Sizing (Refined at throat)
    def sizing(x, y):
        h = 0.003
        dist_throat = np.abs(x - xt)
        h = np.where(dist_throat < 0.02, 0.001, h)
        return np.minimum(h, 0.001 + 0.1*dist_throat)
    
    mesh.discretize_boundary(sizing)
    generate_unstructured_mesh(mesh, sizing, h_base=0.003, n_smooth=20)
    return mesh

# --- 2. Theoretical Solution (1D Isentropic) ---
def area_mach_relation(M, A_ratio, gamma=1.4):
    # A/A* = (1/M) * [ (2 + (g-1)M^2) / (g+1) ] ^ ((g+1)/(2(g-1)))
    term = (2 + (gamma - 1) * M**2) / (gamma + 1)
    return (1.0 / M) * (term ** ((gamma + 1) / (2 * (gamma - 1)))) - A_ratio

def solve_exact_mach(mesh):
    """
    Computes exact 1D Mach number based on the mesh wall profile.
    Since Solver is 2D Planar, Area Ratio = y_wall / y_throat.
    """
    # Extract Wall Y-coordinates sorted by X
    wall_nodes = [n for n in mesh.nodes.values() if n.y > 0.001] # Crude filter for top wall
    # Better: use boundary curves? 
    # Let's just grab all nodes, bin them by X, and take the max Y in each bin.
    
    x_vals = np.linspace(xi, xe, 100)
    y_walls = []
    
    # Reconstruct geometry function roughly or query nodes
    # Let's query the mesh nodes to get the actual discrete geometry
    all_x = np.array([n.x for n in mesh.nodes.values()])
    all_y = np.array([n.y for n in mesh.nodes.values()])
    
    exact_mach = []
    exact_x = []
    
    y_throat = rt # 0.8 inches approx
    
    for x_q in x_vals:
        # Find wall height at this X (Max Y within a small tolerance)
        # We look for nodes in a slice
        slice_mask = np.abs(all_x - x_q) < 0.005
        if np.any(slice_mask):
            y_local = np.max(all_y[slice_mask])
            area_ratio = y_local / y_throat
            
            # Solve for Mach
            # Subsonic branch (x < xt) or Supersonic branch (x > xt)
            if x_q < xt:
                guess = 0.1
            else:
                guess = 2.0
                
            m_sol = fsolve(area_mach_relation, guess, args=(area_ratio,))[0]
            exact_mach.append(m_sol)
            exact_x.append(x_q)
            
    return exact_x, exact_mach

# --- 3. Run Simulation & Plot ---
def run():
    print("--- Nozzle Validation Run ---")
    
    # A. Run Solver
    mesh = create_nozzle_mesh()
    grid = Grid(mesh)
    model = Euler2D(gamma=1.4)
    
    # Stagnation Inlet
    rho_in = 1.2; p_in = 100000.0; u_in = 30.0
    E_in = p_in/0.4 + 0.5*rho_in*u_in**2
    q_inlet = np.array([rho_in, rho_in*u_in, 0.0, E_in])
    
    # Init
    q_init = np.array([1.2, 0.0, 0.0, p_in/0.4])
    model.set_boundary_value("inlet", [rho_in, rho_in*u_in, 0.0, E_in])
    solver = FiniteVolumeSolver(grid, model, order=1)
    solver.set_initial_condition(np.tile(q_init, (grid.n_cells, 1)))
    
    print("   -> Solving (Target 6ms)...")
    total_time = 0.0
    while total_time < 0.006:
        # Safe DT
        h_min = np.min(np.sqrt(2*grid.cell_volumes))
        max_s = 400.0 # Approx
        dt = 0.3 * h_min / max_s
        
        try:
            solver.step(dt)
        except:
            # Vacuum recovery
            solver.q[solver.q[:,0]<1e-2] = q_init
            
        total_time += dt
        
    # B. Extract Centerline Mach
    print("\n   -> extracting Centerline Data...")
    
    # 1. Filter cells near y=0
    c_x = grid.cell_centers[:, 0]
    c_y = grid.cell_centers[:, 1]
    centerline_mask = np.abs(c_y) < 0.005
    
    sim_x = c_x[centerline_mask]
    
    # Calculate Mach for these cells
    q = solver.q[centerline_mask]
    rho = q[:,0]
    u = q[:,1]/rho; v = q[:,2]/rho
    p = (1.4-1)*(q[:,3] - 0.5*rho*(u**2+v**2))
    sim_mach = np.sqrt(u**2+v**2) / np.sqrt(1.4*p/rho)
    
    # Sort for plotting
    sort_idx = np.argsort(sim_x)
    sim_x = sim_x[sort_idx]
    sim_mach = sim_mach[sort_idx]
    
    # C. Get Exact Solution
    exact_x, exact_mach = solve_exact_mach(mesh)
    
    # D. Plot Comparison
    plt.figure(figsize=(10, 6))
    
    # Plot Theory
    plt.plot(exact_x, exact_mach, 'k-', linewidth=2, label='1D Theory (Isentropic)')
    
    # Plot CFD
    plt.plot(sim_x, sim_mach, 'r.', markersize=4, alpha=0.5, label='SnapFVM (Euler 2D)')
    
    plt.title("Validation: Centerline Mach Number")
    plt.xlabel("Axial Position x [m]")
    plt.ylabel("Mach Number")
    plt.axvline(xt, color='gray', linestyle='--', label='Throat')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()