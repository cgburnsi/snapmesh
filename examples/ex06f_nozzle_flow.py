"""
ex06f_nozzle_flow.py
--------------------
Compressible Flow through a Converging-Diverging Rocket Nozzle.
Features:
  1. Unstructured Mesh Generation (SnapMesh).
  2. Refinement at the Throat.
  3. Robust Finite Volume Solver (Numba + Safe Mode).
  4. Artifact-Free Plotting (Maps data to nodes).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# --- HARD RESET ---
# Forces reload of kernels to prevent caching errors
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

def create_nozzle_mesh():
    """
    Defines geometry and generates unstructured mesh.
    """
    # --- 1. Geometry Parameters (from ex03g) ---
    xi  = snapcore.units.convert(0.31, 'inch', 'm')
    ri  = snapcore.units.convert(2.50, 'inch', 'm')
    rci = snapcore.units.convert(0.80, 'inch', 'm')
    rt  = snapcore.units.convert(0.80, 'inch', 'm')
    rct = snapcore.units.convert(0.50, 'inch', 'm')
    xe  = snapcore.units.convert(4.05, 'inch', 'm')
    
    ani = np.deg2rad(44.88)
    ane = np.deg2rad(15.0)

    # Key Points
    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt = xt1 + rct * np.sin(ani)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)

    mesh = Mesh()
    
    # --- 2. Define Curves ---
    mesh.add_curve(LineSegment([xi, 0.0], [xi, ri], name="inlet"))
    
    arc1 = Arc((xi, ri-rci), rci, np.pi/2, np.pi/2 - ani, name="wall")
    mesh.add_curve(arc1)
    
    conv = LineSegment(arc1.evaluate(1.0), [xt1, rt1], name="wall")
    mesh.add_curve(conv)
    
    arc2 = Arc((xt, rt+rct), rct, 3*np.pi/2 - ani, 3*np.pi/2 + ane, name="wall")
    mesh.add_curve(arc2)
    
    div = LineSegment(arc2.evaluate(1.0), [xe, re], name="wall")
    mesh.add_curve(div)
    
    mesh.add_curve(LineSegment([xe, re], [xe, 0.0], name="outlet"))
    
    # Centerline (Symmetry)
    mesh.add_curve(LineSegment([xe, 0.0], [xi, 0.0], name="wall"))
    
    # --- 3. Sizing Function ---
    def sizing(x, y):
        h = 0.003 # 3mm Base
        dist_throat = np.abs(x - xt)
        # Refine to 1mm near throat
        h = np.where(dist_throat < 0.02, 0.001, h)
        # Smooth transition
        h = np.minimum(h, 0.001 + 0.1*dist_throat)
        return h
    
    print("   -> Discretizing Boundary...")
    mesh.discretize_boundary(sizing)
    
    print("   -> Generating Mesh (this may take a moment)...")
    generate_unstructured_mesh(mesh, sizing, h_base=0.003, n_smooth=20)
    return mesh

def compute_safe_dt(grid, solver, cfl=0.3):
    h_local = np.sqrt(2.0 * grid.cell_volumes)
    h_min = np.min(h_local)
    
    rho = np.maximum(solver.q[:,0], 1e-6)
    u = solver.q[:,1] / rho
    v = solver.q[:,2] / rho
    p = (1.4-1.0) * (solver.q[:,3] - 0.5 * rho * (u**2 + v**2))
    p = np.maximum(p, 1e-6)
    c = np.sqrt(1.4 * p / rho)
    
    max_speed = np.max(np.sqrt(u**2 + v**2) + c)
    dt = cfl * (h_min / (max_speed + 1e-6))
    return dt, max_speed

def plot_solution(mesh, grid, solver):
    """
    Plots the solution by mapping Cell Centers -> Mesh Nodes.
    This strictly respects the mesh boundary, removing artifacts.
    """
    print("\n--- Generating Visualization ---")
    
    # 1. Calculate Cell Primitives
    rho = np.maximum(solver.q[:,0], 1e-4)
    u = solver.q[:,1]/rho
    v = solver.q[:,2]/rho
    p = (1.4-1)*(solver.q[:,3] - 0.5*rho*(u**2+v**2))
    p = np.maximum(p, 1e-4)
    mach = np.sqrt(u**2 + v**2) / np.sqrt(1.4 * p / rho)
    
    # 2. Map Cell Data to Nodes (Averaging)
    # This dictionary will store a list of values for each Node ID
    node_mach_acc = {n_id: [] for n_id in mesh.nodes}
    
    # Grid cells are indexed 0..N-1. They correspond to mesh.cells values (if ordered).
    # Since unstructured_gen inserts sequentially, the order *should* match.
    # To be safe, let's iterate the grid cells. But Grid doesn't store Node IDs easily.
    # Strategy: Iterate the MESH cells, assuming they match the GRID order.
    # (Standard behavior of Grid(mesh))
    
    idx = 0
    for c in mesh.cells.values():
        val = mach[idx]
        node_mach_acc[c.n1.id].append(val)
        node_mach_acc[c.n2.id].append(val)
        node_mach_acc[c.n3.id].append(val)
        idx += 1
        
    # 3. Create Arrays for Matplotlib
    # We need a stable index map (0..N_nodes)
    sorted_nodes = sorted(mesh.nodes.values(), key=lambda n: n.id)
    node_map = {n.id: i for i, n in enumerate(sorted_nodes)}
    
    x_nodes = []
    y_nodes = []
    z_mach = []
    
    for n in sorted_nodes:
        x_nodes.append(n.x)
        y_nodes.append(n.y)
        vals = node_mach_acc[n.id]
        if vals:
            z_mach.append(sum(vals) / len(vals))
        else:
            z_mach.append(0.0) # Orphan node (shouldn't happen)

    # 4. Construct Connectivity
    triangles = []
    for c in mesh.cells.values():
        n1 = node_map[c.n1.id]
        n2 = node_map[c.n2.id]
        n3 = node_map[c.n3.id]
        triangles.append([n1, n2, n3])
        
    # 5. Plot
    triang = tri.Triangulation(x_nodes, y_nodes, triangles)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    img1 = ax1.tripcolor(triang, z_mach, cmap='turbo', shading='gouraud')
    ax1.set_title("Nozzle Flow - Mach Number")
    ax1.set_xlabel("Axial Position [m]")
    ax1.set_ylabel("Radial Position [m]")
    ax1.set_aspect('equal')
    plt.colorbar(img1, ax=ax1, label="Mach")
    
    plt.tight_layout()
    plt.show()

def run():
    print("--- Rocket Nozzle Simulation (Euler) ---")
    
    # 1. Mesh
    mesh = create_nozzle_mesh()
    grid = Grid(mesh)
    print(f"   -> Grid: {grid.n_cells} Cells")
    
    # 2. Physics
    model = Euler2D(gamma=1.4)
    
    # Initial Condition: Stagnation
    rho_0 = 1.2
    p_0 = 100000.0
    u_0 = 0.0
    E_0 = p_0/0.4
    q_init = np.array([rho_0, 0, 0, E_0])
    
    # BCs
    rho_in = 1.2
    u_in = 30.0 # ~Mach 0.1 start
    p_in = 100000.0
    E_in = p_in/0.4 + 0.5*rho_in*u_in**2
    model.set_boundary_value("inlet", [rho_in, rho_in*u_in, 0.0, E_in])
    
    solver = FiniteVolumeSolver(grid, model, order=1)
    solver.set_initial_condition(np.tile(q_init, (grid.n_cells, 1)))
    
    # 3. Solve
    total_time = 0.0
    target_time = 0.005 # 5ms
    iter_count = 0
    
    print(f"   -> Starting Solver...")
    
    while total_time < target_time:
        dt, max_s = compute_safe_dt(grid, solver, cfl=0.3)
        
        try:
            resid = solver.step(dt)
        except Exception as e:
            print(f"Error: {e}")
            break
            
        # Vacuum Recovery
        bad_cells = solver.q[:, 0] < 1e-2
        if np.any(bad_cells):
            solver.q[bad_cells] = q_init 
            
        total_time += dt
        iter_count += 1
        
        if iter_count % 500 == 0:
            print(f"   Iter {iter_count:5d}: Time={total_time:.4f}s, dt={dt:.2e}, Resid={resid:.2e}, MaxS={max_s:.1f}")
            if iter_count > 20000: break

    # Pass MESH to plotting
    plot_solution(mesh, grid, solver)

if __name__ == "__main__":
    run()