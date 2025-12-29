import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Project Imports
import ex11 as setup
import snapfvm.solver as fvm_solver

def solve_isentropic_mach(area_ratio, gamma=1.4, regime='subsonic'):
    """
    Solves A/A* = f(M).
    regime: 'subsonic' (guesses M=0.1) or 'supersonic' (guesses M=2.0)
    """
    def eq(M):
        term1 = (2 + (gamma - 1) * M**2) / (gamma + 1)
        term2 = (gamma + 1) / (2 * (gamma - 1))
        return (1/M) * (term1 ** term2) - area_ratio

    guess = 0.1 if regime == 'subsonic' else 2.0
    return fsolve(eq, guess)[0]

def run_simulation_and_get_data():
    """Runs the solver (assuming you want to re-run it)"""
    print("--- 1. Setting Up Simulation ---")
    grid, field = setup.setup_simulation()
    solver = fvm_solver.EulerSolver(grid, field)
    
    print("--- 2. Running Solver ---")
    target_time = 0.0018
    current_time = 0.0
    iter_count = 0
    
    while current_time < target_time:
        dt = solver.compute_time_step(cfl=0.5)
        if current_time + dt > target_time: dt = target_time - current_time
        solver.compute_fluxes()
        solver.update_field(dt)
        current_time += dt
        iter_count += 1
        if iter_count % 1000 == 0:
            print(f"    Iter {iter_count} | t={current_time*1000:.1f}ms")
            
    print("--- Simulation Complete ---")
    return grid, field

def extract_wall_profile(grid):
    tags = np.array(grid.face_tags, dtype=object)
    mask_wall = (tags == "Top") | (tags == "Wall")
    x_wall = grid.face_centers_x[mask_wall]
    y_wall = grid.face_centers_y[mask_wall]
    
    # Sort by X
    idx = np.argsort(x_wall)
    return x_wall[idx], y_wall[idx]

def validate():
    # 1. Run CFD
    grid, field = run_simulation_and_get_data()
    
    # 2. Extract CFD Centerline
    mask_cl = grid.cell_centers_y < 0.005
    x_cfd = grid.cell_centers_x[mask_cl]
    
    c_cfd = np.sqrt(field.gamma * field.p[mask_cl] / field.rho[mask_cl])
    vel_cfd = np.sqrt(field.u[mask_cl]**2 + field.v[mask_cl]**2)
    m_cfd = vel_cfd / c_cfd
    
    idx = np.argsort(x_cfd)
    x_cfd = x_cfd[idx]
    m_cfd = m_cfd[idx]
    
    # 3. Calculate Theory
    x_wall, r_wall = extract_wall_profile(grid)
    area_wall = np.pi * r_wall**2
    
    # Find Throat Index
    idx_throat = np.argmin(area_wall)
    a_star = area_wall[idx_throat]
    
    m_theory = []
    for i, a in enumerate(area_wall):
        ar = a / a_star
        # SWITCH LOGIC: Subsonic before throat, Supersonic after
        if i < idx_throat:
            m_val = solve_isentropic_mach(ar, regime='subsonic')
        else:
            m_val = solve_isentropic_mach(ar, regime='supersonic')
        m_theory.append(m_val)
        
    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_cfd, m_cfd, 'b-', linewidth=2, label='CFD Centerline (Axisymmetric)')
    plt.plot(x_wall, m_theory, 'r--', linewidth=2, label='Theoretical Isentropic Flow')
    
    plt.title("Validation: CFD vs. Theory")
    plt.xlabel("Axial Position [m]")
    plt.ylabel("Mach Number")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.axvline(x_wall[idx_throat], color='k', linestyle=':', alpha=0.5, label='Throat')
    plt.show()

if __name__ == "__main__":
    validate()