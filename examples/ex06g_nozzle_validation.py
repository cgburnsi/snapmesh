"""
ex06g_nozzle_validation.py
--------------------------
Validates the Nozzle Simulation against 1D Isentropic Theory.
FIXED: Uses Dynamic CFL to handle increasing flow speeds (Mach 0 -> Mach 3).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- HARD RESET ---
modules_to_kill = ['snapfvm.numerics', 'snapfvm.solver', 'snapfvm.physics.euler']
for mod in modules_to_kill:
    if mod in sys.modules:
        del sys.modules[mod]

import snapcore.units as cv
from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment, Arc
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.euler import Euler2D

# --- 1. Geometry Setup ---
xi  = snapcore.units.convert(0.31, 'inch', 'm')
ri  = snapcore.units.convert(2.50, 'inch', 'm')
rci = snapcore.units.convert(0.80, 'inch', 'm')
rt  = snapcore.units.convert(0.80, 'inch', 'm')
rct = snapcore.units.convert(0.50, 'inch', 'm')
xe  = snapcore.units.convert(4.05, 'inch', 'm')
ani = np.deg2rad(44.88)
ane = np.deg2rad(15.0)

# Throat X location (needed for theory calc)
xtan = xi + rci * np.sin(ani)
rtan = ri + rci * (np.cos(ani) - 1.0)
rt1 = rt - rct * (np.cos(ani) - 1.0)
xt1 = xtan + (rtan - rt1) / np.tan(ani)
xt = xt1 + rct * np.sin(ani)

def create_nozzle_mesh():
    mesh = Mesh()
    mesh.add_curve(LineSegment([xi, 0.0], [xi, ri], name="inlet"))
    
    # Walls
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
    
    # Refined Throat Sizing
    def sizing(x, y):
        h = 0.003
        dist_throat = np.abs(x - xt)
        h = np.where(dist_throat < 0.02, 0.001, h)
        return np.minimum(h, 0.001 + 0.1*dist_throat)
    
    mesh.discretize_boundary(sizing)
    generate_unstructured_mesh(mesh, sizing, h_base=0.003, n_smooth=20)
    return mesh

# --- 2. Theory (1D Isentropic Flow) ---
def area_mach_relation(M, A_ratio, gamma=1.4):
    if M <= 0: return 1e9
    term = (2 + (gamma - 1) * M**2) / (gamma + 1)
    # Eq: A/A* = (1/M) * [ ... ] ^ ...
    return (1.0 / M) * (term ** ((gamma + 1) / (2 * (gamma - 1)))) - A_ratio

def solve_exact_mach(mesh):
    """
    Scans the mesh geometry to build the theoretical Area Ratio curve.
    """
    print("   -> Calculating Theoretical Solution...")
    all_x = np.array([n.x for n in mesh.nodes.values()])
    all_y = np.array([n.y for n in mesh.nodes.values()])
    
    # Discretize X axis
    x_theory = np.linspace(xi + 0.001, xe - 0.001, 100)
    mach_theory = []
    
    y_throat = rt
    
    for x_q in x_theory:
        # Find wall height at this X
        # Get nodes within a thin slice
        mask = np.abs(all_x - x_q) < 0.005
        if np.any(mask):
            y_wall = np.max(all_y[mask])
            A_ratio = y_wall / y_throat
            
            # Solve M for this Area Ratio
            # Subsonic branch if x < xt, Supersonic if x > xt
            guess = 0.1 if x_q < xt else 2.5
            try:
                m_sol = fsolve(area_mach_relation, guess, args=(A_ratio,))[0]
            except:
                m_sol = guess
            mach_theory.append(m_sol)
        else:
            mach_theory.append(np.nan) # Should not happen
            
    return x_theory, mach_theory

# --- 3. Dynamic Time Stepping ---
def compute_dt(grid, solver, cfl=0.4):
    h_min = np.min(np.sqrt(2 * grid.cell_volumes))
    
    # Calculate Max Wave Speed (u + c)
    rho = np.maximum(solver.q[:,0], 1e-6)
    u = solver.q[:,1] / rho
    v = solver.q[:,2] / rho
    p = (1.4-1.0) * (solver.q[:,3] - 0.5 * rho * (u**2 + v**2))
    p = np.maximum(p, 1e-6)
    c = np.sqrt(1.4 * p / rho)
    
    max_s = np.max(np.sqrt(u**2 + v**2) + c)
    
    # dt = CFL * h / max_speed
    return cfl * h_min / (max_s + 1e-6)

# --- 4. Main Run ---
def run():
    print("--- Nozzle Validation (Robust) ---")
    
    mesh = create_nozzle_mesh()
    grid = Grid(mesh)
    print(f"   -> Grid: {grid.n_cells} cells")
    
    model = Euler2D(gamma=1.4)
    
    # Inlet Conditions
    rho_in = 1.2
    p_in = 100000.0
    u_in = 30.0 # Low speed inflow
    E_in = p_in/0.4 + 0.5*rho_in*u_in**2
    
    q_inlet = np.array([rho_in, rho_in*u_in, 0.0, E_in])
    q_init = np.array([1.2, 0.0, 0.0, p_in/0.4]) # Stagnation start
    
    model.set_boundary_value("inlet", q_inlet)
    solver = FiniteVolumeSolver(grid, model, order=1)
    solver.set_initial_condition(np.tile(q_init, (grid.n_cells, 1)))
    
    print("   -> Starting Solver (Target 6ms)...")
    total_time = 0.0
    target_time = 0.006
    iter_count = 0
    
    while total_time < target_time:
        # Dynamic DT is critical here
        dt = compute_dt(grid, solver, cfl=0.4)
        
        try:
            resid = solver.step(dt)
        except Exception as e:
            print(f"Solver Crash: {e}")
            break
            
        # Vacuum Recovery
        if np.any(solver.q[:,0] < 1e-2):
            solver.q[solver.q[:,0] < 1e-2] = q_init
            
        total_time += dt
        iter_count += 1
        
        if iter_count % 500 == 0:
            print(f"   Iter {iter_count:4d}: t={total_time*1000:.2f}ms, dt={dt:.2e}, Resid={resid:.2e}")
            
    # --- Visualization ---
    print("\n   -> Plotting Validation...")
    
    # 1. Extract Centerline Data
    # Filter cells near Y=0
    cy = grid.cell_centers[:, 1]
    cx = grid.cell_centers[:, 0]
    mask = np.abs(cy) < 0.005 # 5mm tolerance
    
    sim_x = cx[mask]
    q_sim = solver.q[mask]
    
    rho = np.maximum(q_sim[:,0], 1e-6)
    u = q_sim[:,1]/rho
    v = q_sim[:,2]/rho
    p = (1.4-1)*(q_sim[:,3] - 0.5*rho*(u**2+v**2))
    p = np.maximum(p, 1e-6)
    sim_mach = np.sqrt(u**2+v**2) / np.sqrt(1.4*p/rho)
    
    # Sort for plotting line
    idx = np.argsort(sim_x)
    sim_x = sim_x[idx]
    sim_mach = sim_mach[idx]
    
    # 2. Get Theory
    theory_x, theory_mach = solve_exact_mach(mesh)
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(theory_x, theory_mach, 'k-', linewidth=2, label="1D Isentropic Theory")
    plt.plot(sim_x, sim_mach, 'r.', markersize=4, alpha=0.5, label="CFD Result (Euler)")
    
    plt.axvline(xt, color='gray', linestyle='--', label="Throat")
    plt.xlabel("Axial Position [m]")
    plt.ylabel("Mach Number")
    plt.title("Nozzle Validation: CFD vs Theory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()