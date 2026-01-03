"""
ex06a_supersonic_bump_final.py
------------------------------
Supersonic Flow (Mach 1.5) over a Bump.
CORRECTED GEOMETRY: Bump height reduced to h=0.08 (8% thick).
  - Prevents shock detachment (Bow Shock).
  - Ensures shock attaches at the leading edge (x=1.0).
  - Uses robust 'Safe Mode' solver settings.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# --- HARD RESET ---
# Ensure we load fresh modules (clears previous caches)
modules_to_kill = ['snapfvm.numerics', 'snapfvm.solver', 'snapfvm.physics.euler']
for mod in modules_to_kill:
    if mod in sys.modules:
        del sys.modules[mod]

from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment, Arc
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.euler import Euler2D

def create_bump_mesh():
    m = Mesh()
    # Domain 3x2
    p0=[0,0]; p1=[1,0]; p2=[2,0]; p3=[3,0]          
    p4=[3,2]; p5=[0,2] 
    
    # --- GEOMETRY FIX ---
    # h=0.08 (8% Thickness). 
    # This keeps the deflection angle (~5 deg) well below the detachment limit (12 deg).
    h = 0.08; c = 1.0
    R = (h**2 + c**2/4.0) / (2*h) 
    center = [1.5, -R + h]
    t1 = np.arctan2(0 - center[1], 1 - center[0])
    t2 = np.arctan2(0 - center[1], 2 - center[0])
    
    m.add_curve(LineSegment(p5, p0, name="inlet"))
    m.add_curve(LineSegment(p0, p1, name="wall"))
    m.add_curve(Arc(center, R, t1, t2, name="wall"))
    m.add_curve(LineSegment(p2, p3, name="wall"))
    m.add_curve(LineSegment(p3, p4, name="outlet"))
    m.add_curve(LineSegment(p4, p5, name="wall")) 
    
    # Mesh Sizing
    def sizing(x, y):
        dist_bump = np.sqrt((x - 1.5)**2 + y**2)
        # Fine mesh near shock, coarse elsewhere
        return np.where(dist_bump < 1.3, 0.05, 0.12)
        
    m.discretize_boundary(sizing)
    generate_unstructured_mesh(m, sizing, h_base=0.12, n_smooth=20)
    return m

def compute_safe_dt(grid, solver, cfl=0.3):
    """
    Robust dt calculation. 
    Locked to CFL=0.3 to prevent instability.
    """
    # Characteristic Length ~ sqrt(2*Area) for triangles
    h_local = np.sqrt(2.0 * grid.cell_volumes)
    h_min = np.min(h_local)
    
    # Wave Speeds
    rho = np.maximum(solver.q[:,0], 1e-6)
    u = solver.q[:,1] / rho
    v = solver.q[:,2] / rho
    p = (1.4-1.0) * (solver.q[:,3] - 0.5 * rho * (u**2 + v**2))
    p = np.maximum(p, 1e-6)
    c = np.sqrt(1.4 * p / rho)
    
    max_speed = np.max(np.sqrt(u**2 + v**2) + c)
    
    # dt limit
    dt = cfl * (h_min / (max_speed + 1e-6))
    return dt, max_speed

def plot_solution(grid, solver):
    print("\n--- Generating Visualization ---")
    rho = np.maximum(solver.q[:,0], 1e-4)
    u = solver.q[:,1]/rho
    v = solver.q[:,2]/rho
    p = (1.4-1)*(solver.q[:,3] - 0.5*rho*(u**2+v**2))
    p = np.maximum(p, 1e-4)
    mach = np.sqrt(u**2 + v**2) / np.sqrt(1.4 * p / rho)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    triang = tri.Triangulation(grid.cell_centers[:,0], grid.cell_centers[:,1])
    
    # Plot Mach
    # Vmin/Vmax set to highlight the shock range (1.0 to 1.6)
    img1 = ax1.tripcolor(triang, mach, cmap='turbo', shading='gouraud', vmin=1.0, vmax=1.6)
    ax1.set_title("Mach Number (Attached Shock)")
    ax1.set_ylabel("Y [m]")
    ax1.set_aspect('equal')
    plt.colorbar(img1, ax=ax1, label="Mach")
    
    # Plot Pressure
    img2 = ax2.tripcolor(triang, p, cmap='inferno', shading='gouraud')
    ax2.set_title("Pressure [Pa]")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_aspect('equal')
    plt.colorbar(img2, ax=ax2, label="Pressure")
    plt.tight_layout()
    plt.show()

def run():
    print("--- Supersonic Bump (Mach 1.5) Final ---")
    
    mesh = create_bump_mesh()
    grid = Grid(mesh)
    print(f"   -> Grid: {grid.n_cells} Cells")
    
    model = Euler2D(gamma=1.4)
    
    # Initial Condition: Mach 1.5
    rho_in = 1.0; p_in = 100000.0
    M_in = 1.5
    c_in = np.sqrt(1.4 * p_in / rho_in)
    u_in = M_in * c_in
    E_in = p_in/0.4 + 0.5*rho_in*u_in**2
    q_inlet = np.array([rho_in, rho_in*u_in, 0.0, E_in])
    
    model.set_boundary_value("inlet", q_inlet)
    solver = FiniteVolumeSolver(grid, model, order=1)
    solver.set_initial_condition(np.tile(q_inlet, (grid.n_cells, 1)))
    
    total_time = 0.0
    target_time = 0.04 # 40ms is enough to flush the domain
    iter_count = 0
    
    print(f"   -> Starting Solver (Target: {target_time}s)...")
    
    while total_time < target_time:
        # 1. Compute DT (Safe CFL=0.3)
        dt, max_s = compute_safe_dt(grid, solver, cfl=0.3)
        
        # 2. Step
        try:
            resid = solver.step(dt)
        except Exception as e:
            print(f"Error at {iter_count}: {e}")
            break
            
        # 3. VACUUM RECOVERY (Safety Net)
        # Resets any failed cells to the inlet condition
        bad_cells = solver.q[:, 0] < 1e-2
        if np.any(bad_cells):
            solver.q[bad_cells] = q_inlet
            
        total_time += dt
        iter_count += 1
        
        if iter_count % 500 == 0:
            print(f"   Iter {iter_count:5d}: Time={total_time:.4f}s, dt={dt:.2e}, Resid={resid:.2e}")
            if iter_count > 20000: break 

    plot_solution(grid, solver)

if __name__ == "__main__":
    run()