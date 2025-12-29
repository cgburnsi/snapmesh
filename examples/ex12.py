import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Project Imports
import ex11 as setup  # Reuse the geometry/setup from previous file
import snapfvm.solver as fvm_solver

def run_solver():
    # 1. Setup
    print("--- Initializing ---")
    grid, field = setup.setup_simulation()
    
    # 2. Create Solver
    solver = fvm_solver.EulerSolver(grid, field)
    
    # 3. Time Loop
    target_time = 0.002 # Run Time
    current_time = 0.0
    iteration = 0
    
    print("--- Starting Time Marching ---")
    while current_time < target_time:
        # A. Calculate dt
        dt = solver.compute_time_step(cfl=0.3) # Conservative CFL start
        
        # Don't overshoot target time
        if current_time + dt > target_time:
            dt = target_time - current_time
            
        # B. Compute Fluxes
        solver.compute_fluxes()
        
        # C. Update State
        solver.update_field(dt)
        
        current_time += dt
        iteration += 1
        
        if iteration % 100 == 0:
            # Calculate Mach for reporting
            c = np.sqrt(field.gamma * field.p / field.rho)
            vel = np.sqrt(field.u**2 + field.v**2)
            mach = vel / c
            print(f"Iter {iteration:5d} | Time {current_time*1000:.2f} ms | dt {dt:.2e} | Max Mach {np.max(mach):.2f}")

    print("--- Simulation Complete ---")
    
    # 4. Plot Final Result (Mach Number)
    c = np.sqrt(field.gamma * field.p / field.rho)
    vel = np.sqrt(field.u**2 + field.v**2)
    mach = vel / c
    
    fig, ax = plt.subplots(figsize=(12, 6))
    triang = mtri.Triangulation(grid.nodes_x, grid.nodes_y, grid.cell_nodes)
    
    contour = ax.tripcolor(triang, mach, cmap='inferno', shading='flat')
    cbar = fig.colorbar(contour, label='Mach Number')
    
    ax.set_aspect('equal')
    ax.set_title(f"Nozzle Flow Mach Number (t={current_time*1000:.2f} ms)")
    plt.show()

if __name__ == '__main__':
    run_solver()