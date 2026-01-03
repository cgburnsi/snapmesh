"""
ex10b_rocket_nozzle.py
----------------------
Compressible Flow Application: DeLaval Nozzle.
Config: Euler 2D | Second-Order | Stagnation Inlet.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- HARD RESET ---
modules_to_kill = ['snapfvm.numerics', 'snapfvm.solver', 'snapfvm.physics.euler', 'snapfvm.display', 'snapcore.units']
for mod in modules_to_kill:
    if mod in sys.modules:
        del sys.modules[mod]

# --- IMPORTS ---
import snapcore.units as cv               # <--- FIXED IMPORT
from snapcore.display import SimulationDisplay
from snapmesh.mesh import Mesh
from snapmesh.geometry import LineSegment, Arc
from snapmesh.unstructured_gen import generate_unstructured_mesh
from snapfvm.grid import Grid
from snapfvm.solver import FiniteVolumeSolver
from snapfvm.physics.euler import Euler2D

# --- GEOMETRY DEFINITION ---
# Uses 'cv.convert' consistently now
xi  = cv.convert(0.31, 'inch', 'm')
ri  = cv.convert(2.50, 'inch', 'm')
rci = cv.convert(0.80, 'inch', 'm')
rt  = cv.convert(0.80, 'inch', 'm')
rct = cv.convert(0.50, 'inch', 'm')
xe  = cv.convert(4.05, 'inch', 'm')
ani = np.deg2rad(44.88)
ane = np.deg2rad(15.0)

xtan = xi + rci * np.sin(ani)
rtan = ri + rci * (np.cos(ani) - 1.0)
rt1 = rt - rct * (np.cos(ani) - 1.0)
xt1 = xtan + (rtan - rt1) / np.tan(ani)
xt = xt1 + rct * np.sin(ani)

def create_nozzle_mesh():
    mesh = Mesh()
    mesh.add_curve(LineSegment([xi, 0.0], [xi, ri], name="inlet"))
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
    mesh.add_curve(LineSegment([xe, 0.0], [xi, 0.0], name="wall")) 
    
    def sizing(x, y):
        h = 0.003
        dist_throat = np.abs(x - xt)
        h = np.where(dist_throat < 0.02, 0.001, h)
        return np.minimum(h, 0.001 + 0.1*dist_throat)
    
    mesh.discretize_boundary(sizing)
    generate_unstructured_mesh(mesh, sizing, h_base=0.003, n_smooth=20)
    return mesh

# --- PHYSICS HELPERS ---
def update_inlet_bc(solver, grid, p0, rho0, inlet_face_indices):
    """Subsonic Stagnation Inlet BC"""
    gamma = solver.model.gamma
    interior_cells = grid.face_cells[inlet_face_indices, 0].astype(int)
    q_int = solver.q[interior_cells]
    
    rho = np.maximum(q_int[:,0], 1e-6)
    u = q_int[:,1]/rho
    v = q_int[:,2]/rho
    p_int = (gamma-1)*(q_int[:,3] - 0.5*rho*(u**2+v**2))
    p_int = np.maximum(p_int, 1e-4)
    
    p_face = np.minimum(p_int, p0 - 1e-4)
    M2 = (2.0/(gamma-1.0)) * ( (p0/p_face)**((gamma-1.0)/gamma) - 1.0 )
    M = np.sqrt(np.maximum(M2, 0.0))
    rho_face = rho0 * (p_face/p0)**(1.0/gamma)
    c_face = np.sqrt(gamma * p_face / rho_face)
    
    vel_mag = M * c_face
    avg_rho = np.mean(rho_face)
    avg_u   = np.mean(vel_mag)
    avg_E   = np.mean(p_face/(gamma-1.0) + 0.5*rho_face*vel_mag**2)
    
    solver.model.set_boundary_value("inlet", [avg_rho, avg_rho*avg_u, 0.0, avg_E])

def compute_dt(grid, solver, cfl=0.4):
    h_min = np.min(np.sqrt(2 * grid.cell_volumes))
    rho = np.maximum(solver.q[:,0], 1e-6)
    u = solver.q[:,1] / rho
    v = solver.q[:,2] / rho
    p = (1.4-1.0) * (solver.q[:,3] - 0.5 * rho * (u**2 + v**2))
    p = np.maximum(p, 1e-6)
    c = np.sqrt(1.4 * p / rho)
    max_s = np.max(np.sqrt(u**2 + v**2) + c)
    return cfl * h_min / (max_s + 1e-6), max_s

# --- MAIN EXECUTION ---
def run():
    disp = SimulationDisplay("Rocket Nozzle Simulation", "Euler 2D | Order=2 | Stagnation Inlet")
    disp.header()
    
    disp.section("Mesh Generation")
    mesh = create_nozzle_mesh()
    grid = Grid(mesh)
    print(f"   -> Generated {grid.n_cells} unstructured cells")
    
    model = Euler2D(gamma=1.4)
    p0 = 100000.0; rho0 = 1.2
    
    q_init = np.array([rho0, 0.0, 0.0, p0/0.4])
    solver = FiniteVolumeSolver(grid, model, order=2)
    solver.set_initial_condition(np.tile(q_init, (grid.n_cells, 1)))
    
    inlet_group_id = [gid for gid, name in grid.group_names.items() if name == "inlet"][0]
    inlet_faces = np.where(grid.face_groups == inlet_group_id)[0]
    
    disp.section("Time Integration")
    disp.setup_stats_columns(["Iter", "Time [ms]", "dt", "Resid", "Max(u+c)"], [6, 12, 10, 10, 10])
    
    total_time = 0.0; target_time = 0.008; iter_count = 0
    dt = 1e-7
    resid = 1.0; max_s = 340.0
    
    while total_time < target_time:
        if iter_count > 0: update_inlet_bc(solver, grid, p0, rho0, inlet_faces)
        
        target_cfl = 0.4
        cfl = min(target_cfl, 0.1 + iter_count*0.001)
        dt, max_s = compute_dt(grid, solver, cfl=cfl)
        
        try:
            resid = solver.step(dt)
        except Exception as e:
            disp.error(f"Solver Crash: {e}")
            break
            
        if np.any(solver.q[:,0] < 1e-2): solver.q[solver.q[:,0] < 1e-2] = q_init
            
        total_time += dt; iter_count += 1
        if iter_count % 500 == 0:
            disp.log_stats(iter_count, total_time*1000, dt, resid, max_s)
            
    disp.success()

    disp.section("Visualization")
    mask = np.abs(grid.cell_centers[:, 1]) < 0.005
    sim_x = grid.cell_centers[mask, 0]
    q_sim = solver.q[mask]
    
    rho = np.maximum(q_sim[:,0], 1e-6)
    u = q_sim[:,1]/rho; v = q_sim[:,2]/rho
    p = (1.4-1)*(q_sim[:,3] - 0.5*rho*(u**2+v**2))
    p = np.maximum(p, 1e-6)
    sim_mach = np.sqrt(u**2+v**2) / np.sqrt(1.4*p/rho)
    
    idx = np.argsort(sim_x)
    sim_x = sim_x[idx]; sim_mach = sim_mach[idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sim_x, sim_mach, 'r.', label="CFD (Euler 2nd Order)")
    plt.title("Nozzle Validation (Order=2)")
    plt.xlabel("Axial Position [m]"); plt.ylabel("Mach Number")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()

if __name__ == "__main__":
    run()