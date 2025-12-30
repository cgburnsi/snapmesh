import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import snapmesh.unstructured_gen as umesh
import snapfvm.grid as fvm
import snapfvm.field as flow
import snapfvm.solver as fvm_solver
import unit_convert as cv

# --- 1. Exact Geometry Definition (From ex11.py) ---
def get_nozzle_params():
    """Calculates the key geometric points for the nozzle."""
    # Parameters from ex11
    xi  = cv.convert(0.31, 'inch', 'm')
    ri  = cv.convert(2.50, 'inch', 'm')
    rci = cv.convert(0.80, 'inch', 'm')
    rt  = cv.convert(0.80, 'inch', 'm')
    rct = cv.convert(0.50, 'inch', 'm')
    xe  = cv.convert(4.05, 'inch', 'm')
    ani = np.deg2rad(44.88)
    ane = np.deg2rad(15.0)

    # Key Points Calculations
    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt = xt1 + rct * np.sin(ani)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)
    
    return {
        'xi': xi, 'ri': ri, 'xe': xe, 're': re, 'xt': xt, 'rt': rt,
        'rci': rci, 'rct': rct, 'ani': ani, 'ane': ane,
        'xtan': xtan, 'rtan': rtan, 'xt1': xt1, 'rt1': rt1, 'xt2': xt2, 'rt2': rt2
    }

def get_exact_boundary_poly(n_pts=100):
    """Generates the closed polygon loop using the exact analytic curves."""
    p = get_nozzle_params()
    
    # --- 1. Construct Top Wall Segments (Forward: Inlet -> Exit) ---
    
    # Segment A: Inlet Arc (Left -> Right)
    # Angle goes from 90 down to (90-ani)
    theta_in = np.linspace(np.pi/2, np.pi/2 - p['ani'], 20)
    x_a1 = p['xi'] + p['rci'] * np.cos(theta_in)
    y_a1 = (p['ri'] - p['rci']) + p['rci'] * np.sin(theta_in)
    
    # Segment B: Converging Line (Left -> Right)
    x_l1 = np.linspace(p['xtan'], p['xt1'], 20)
    m_conv = -np.tan(p['ani'])
    y_l1 = p['rtan'] + m_conv * (x_l1 - p['xtan'])
    
    # Segment C: Throat Arc (Left -> Right)
    # Angle goes from (270-ani) to (270+ane)
    # Note: 270 deg is 3*pi/2
    t_start = 1.5*np.pi - p['ani']
    t_end   = 1.5*np.pi + p['ane']
    theta_th = np.linspace(t_start, t_end, 30)
    x_a2 = p['xt'] + p['rct'] * np.cos(theta_th)
    y_a2 = (p['rt'] + p['rct']) + p['rct'] * np.sin(theta_th)
    
    # Segment D: Diverging Line (Left -> Right)
    x_l2 = np.linspace(p['xt2'], p['xe'], 30)
    m_div = np.tan(p['ane'])
    y_l2 = p['rt2'] + m_div * (x_l2 - p['xt2'])
    
    # --- 2. Assemble Loop ---
    
    # Bottom: (xi, 0) -> (xe, 0)
    pts_bot = np.column_stack([np.linspace(p['xi'], p['xe'], 50), np.zeros(50)])
    
    # Right: (xe, 0) -> (xe, re)
    pts_out = np.column_stack([np.full(15, p['xe']), np.linspace(0, p['re'], 15)])
    
    # Top: (xe, re) -> (xi, ri)
    # Concatenate FORWARD (Left->Right), then REVERSE to go Right->Left
    top_x = np.concatenate([x_a1, x_l1, x_a2, x_l2])
    top_y = np.concatenate([y_a1, y_l1, y_a2, y_l2])
    
    # Reverse for CCW loop (Exit -> Inlet)
    pts_top = np.column_stack([top_x[::-1], top_y[::-1]])
    
    # Left: (xi, ri) -> (xi, 0)
    pts_in = np.column_stack([np.full(15, p['xi']), np.linspace(p['ri'], 0, 15)])
    
    # Stack them all
    return np.vstack([pts_bot, pts_out, pts_top, pts_in])

# --- 2. Sizing Function (Geometry Aware) ---
def exact_nozzle_sizing(x, y):
    p = get_nozzle_params()
    
    # 1. Base Size (Free stream)
    h_base = 0.005 # 5mm
    
    # 2. Refine at Throat (Axial)
    # Concentrate resolution around x = xt
    dist_throat = np.abs(x - p['xt'])
    # Sigmoid blend: 0.3 factor at throat, 1.0 factor far away
    f_throat = 0.3 + 0.7 * np.tanh(dist_throat / 0.02)
    
    # 3. Refine at Wall (Radial)
    # Approximate wall radius r_wall(x)
    # Simple linear interp is "good enough" for sizing function distance
    # Throat region is most critical
    r_approx = np.interp(x, [p['xi'], p['xt'], p['xe']], [p['ri'], p['rt'], p['re']])
    dist_to_wall = np.abs(r_approx - y)
    
    # Boundary Layer factor: Small h near wall
    # We want h ~ 1mm at wall, blending to h_base inside
    f_wall = 0.2 + 0.8 * np.tanh(dist_to_wall / 0.015) 
    
    return h_base * f_throat * f_wall

# --- 3. Simulation Runner ---
def run_simulation():
    print("1. Generating Exact Geometry Unstructured Mesh...")
    poly = get_exact_boundary_poly()
    points, tris = umesh.generate_unstructured_mesh(poly, exact_nozzle_sizing, h_base=0.005, n_smooth=15)
    print(f"   -> Generated {len(points)} Nodes, {len(tris)} Cells")
    
    print("2. Converting to Solver Grid...")
    mesh_obj = umesh.raw_to_snapmesh(points, tris)
    grid = fvm.UnstructuredGrid(mesh_obj)
    
    print("3. Initializing High-Pressure Flow...")
    sol = flow.Field(grid)
    
    # High Pressure Ratio (approx 5:1)
    p_inlet, p_exit = 500000.0, 100000.0
    x = grid.cell_centers_x
    alpha = (x - np.min(x)) / (np.max(x) - np.min(x))
    p_init = p_inlet*(1-alpha) + p_exit*alpha
    
    sol.rho[:] = p_init / (287.05 * 300.0)
    sol.p[:]   = p_init
    # Approx Mach 0.1 start to avoid shock on boot
    sol.u[:]   = 10.0 
    sol.v[:]   = 0.0
    sol.rhou[:] = sol.rho * sol.u
    sol.rhoE[:] = (sol.p / 0.4) + 0.5 * sol.rho * sol.u**2
    
    print("4. Running Axisymmetric HLLC Solver...")
    solver = fvm_solver.EulerSolver(grid, sol)
    solver.P_stag = 500000.0
    solver.rho_stag = solver.P_stag / (287.05 * 300.0)
    
    t_end = 0.001
    t = 0.0
    i = 0
    while t < t_end:
        dt = solver.compute_time_step(cfl=0.6)
        if t + dt > t_end: dt = t_end - t
        
        solver.compute_fluxes()
        solver.update_field(dt)
        
        t += dt
        i += 1
        if i % 200 == 0:
            mach = np.sqrt(sol.u**2+sol.v**2)/np.sqrt(1.4*sol.p/sol.rho)
            print(f"   Iter {i:5d} | t={t*1000:.2f}ms | Max Mach={np.max(mach):.2f}")
            
    print("5. Plotting...")
    fig, ax = plt.subplots(figsize=(10, 5))
    triang = mtri.Triangulation(grid.nodes_x, grid.nodes_y, grid.cell_nodes)
    mach = np.sqrt(sol.u**2+sol.v**2)/np.sqrt(1.4*sol.p/sol.rho)
    
    contour = ax.tripcolor(triang, mach, cmap='turbo', shading='flat')
    fig.colorbar(contour, label='Mach Number')
    ax.triplot(triang, 'k-', lw=0.1, alpha=0.4) # Show the grid!
    
    ax.set_aspect('equal')
    ax.set_title(f"Unstructured Nozzle (Exact Geometry) - Mach Max={np.max(mach):.2f}")
    plt.show()

if __name__ == "__main__":
    run_simulation()