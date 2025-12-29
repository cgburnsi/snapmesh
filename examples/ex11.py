import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Project Imports
import snapmesh as sm
from snapmesh.transfinite import generate_structured_mesh
import snapfvm.grid as fvm
import snapfvm.field as flow
import unit_convert as cv

# --- 1. Geometry & Mesh Generation ---
def create_structured_nozzle():
    """
    Defines the nozzle geometry and generates a structured mesh.
    """
    # Define Parameters
    xi  = cv.convert(0.31, 'inch', 'm')
    ri  = cv.convert(2.50, 'inch', 'm')
    rci = cv.convert(0.80, 'inch', 'm')
    rt  = cv.convert(0.80, 'inch', 'm')
    rct = cv.convert(0.50, 'inch', 'm')
    xe  = cv.convert(4.05, 'inch', 'm')
    ani = np.deg2rad(44.88)
    ane = np.deg2rad(15.0)

    # Calculate Key Points
    xtan = xi + rci * np.sin(ani)
    rtan = ri + rci * (np.cos(ani) - 1.0)
    rt1 = rt - rct * (np.cos(ani) - 1.0)
    xt1 = xtan + (rtan - rt1) / np.tan(ani)
    xt = xt1 + rct * np.sin(ani)
    xt2 = xt + rct * np.sin(ane)
    rt2 = rt + rct * (1.0 - np.cos(ane))
    re = rt2 + (xe - xt2) * np.tan(ane)

    # Build The 4 Logical Sides
    left_curve   = sm.Line(xi, 0.0, xi, ri)
    right_curve  = sm.Line(xe, 0.0, xe, re)
    bottom_curve = sm.Line(xi, 0.0, xe, 0.0)

    # Top Wall (Composite)
    arc_inlet = sm.Arc(cx=xi, cy=(ri-rci), r=rci, 
                       start_angle=np.deg2rad(90.0), 
                       end_angle=np.deg2rad(90.0) - ani)
    line_conv = sm.Line(xtan, rtan, xt1, rt1)
    arc_throat = sm.Arc(cx=xt, cy=(rt+rct), r=rct,
                        start_angle=np.deg2rad(270.0) - ani,
                        end_angle=np.deg2rad(270.0) + ane)
    line_div = sm.Line(xt2, rt2, xe, re)
    
    top_curve = sm.CompositeCurve([arc_inlet, line_conv, arc_throat, line_div])

    # Generate Mesh (60 streamwise x 15 radial)
    print("  -> Generating structured mesh...")
    mesh = generate_structured_mesh(bottom_curve, top_curve, left_curve, right_curve, ni=60, nj=15)
    
    return mesh

# --- 2. Simulation Setup ---
def setup_simulation():
    # 1. Generate Mesh
    print("Generating Nozzle Mesh...")
    mesh = create_structured_nozzle()
    
    # 2. Build FVM Grid
    print("Building FVM Grid (Connectivity & Metrics)...")
    grid = fvm.UnstructuredGrid(mesh)
    print(f"  -> {grid.num_cells} Cells, {grid.num_faces} Faces")
    
    # 3. Create Flow Field container
    print("Initializing Flow Field...")
    sol = flow.Field(grid)
    
    # 4. Set Initial Conditions
    # Standard Sea Level Inlet -> Vacuum Exit
    p_inlet = cv.convert(70, 'psi', 'Pa') 
    p_exit  = cv.convert(14.7, 'psi', 'Pa')
    T_total = cv.convert(80, 'degF', 'K')
    
    # Linear ramp based on x-coordinate
    x = grid.cell_centers_x
    x_min, x_max = np.min(x), np.max(x)
    alpha = (x - x_min) / (x_max - x_min)
    
    # Apply ramp
    p_init = p_inlet * (1.0 - alpha) + p_exit * alpha
    
    # Initial Density (Ideal Gas)
    rho_init = p_init / (287.05 * T_total)
    
    # Set to Field
    sol.rho[:] = rho_init
    sol.p[:]   = p_init
    sol.u[:]   = 0.0 
    sol.v[:]   = 0.0
    
    # Update Conservative Variables (The ones the solver actually evolves)
    # Energy = p/(gamma-1) + Kinetic
    sol.rhou[:] = 0.0
    sol.rhov[:] = 0.0
    sol.rhoE[:] = (sol.p / (sol.gamma - 1.0)) 
    
    return grid, sol

# --- 3. Visualization ---
def plot_initial_state():
    grid, sol = setup_simulation()
    
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use Tripcolor to show the scalar field on the triangles
    triang = mtri.Triangulation(grid.nodes_x, grid.nodes_y, grid.cell_nodes)
    
    # Plot Pressure
    contour = ax.tripcolor(triang, sol.p, cmap='jet', shading='flat')
    cbar = fig.colorbar(contour, label='Pressure [Pa]')
    
    ax.set_aspect('equal')
    ax.set_title("Initial Pressure Distribution (t=0)")
    ax.set_xlabel("Axial Position [m]")
    ax.set_ylabel("Radial Position [m]")
    
    plt.show()

if __name__ == '__main__':
    plot_initial_state()