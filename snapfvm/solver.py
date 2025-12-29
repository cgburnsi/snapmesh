import numpy as np

class EulerSolver:
    def __init__(self, grid, field):
        self.grid = grid
        self.field = field
        
        # Pre-allocate flux arrays
        self.flux_rho  = np.zeros(grid.num_faces)
        self.flux_rhou = np.zeros(grid.num_faces)
        self.flux_rhov = np.zeros(grid.num_faces)
        self.flux_rhoE = np.zeros(grid.num_faces)

        # --- PRE-PROCESS BOUNDARY MASKS ---
        # We check the string tags once here to speed up the loop
        # Convert list to numpy array for boolean masking
        tags = np.array(grid.face_tags, dtype=object)
        
        # Identify face indices for each type
        # Note: We treat "Centerline" same as "Wall" (Slip condition)
        self.mask_inlet  = (tags == "Left")   # Or "Inlet" depending on ex11 names
        self.mask_outlet = (tags == "Right")  # Or "Outlet"
        self.mask_wall   = (tags == "Top") | (tags == "Bottom") | (tags == "Wall") | (tags == "Centerline")
        self.mask_internal = (grid.face_neighbor != -1)
        
        # Debug print to ensure we found them
        print(f"Solver Init: Found {np.sum(self.mask_inlet)} Inlets, "
              f"{np.sum(self.mask_outlet)} Outlets, {np.sum(self.mask_wall)} Walls")
        
        # Boundary Conditions (Stagnation)
        self.P_stag = 101325.0
        self.T_stag = 300.0
        self.rho_stag = self.P_stag / (287.05 * self.T_stag)
        # Energy at stagnation (u=0) is just internal energy
        self.rhoE_stag = self.P_stag / (field.gamma - 1.0) 

    def compute_time_step(self, cfl=0.5):
        """
        Calculates dt based on CFL condition.
        """
        g = self.grid
        f = self.field
        
        # Sound speed
        c = np.sqrt(f.gamma * np.maximum(f.p, 1e-5) / np.maximum(f.rho, 1e-5))
        vel_mag = np.sqrt(f.u**2 + f.v**2)
        lambda_max = vel_mag + c
        
        length_scale = np.sqrt(g.cell_volumes)
        local_dt = length_scale / (lambda_max + 1e-12)
        
        return cfl * np.min(local_dt)

    def compute_fluxes(self):
        g = self.grid
        f = self.field
        
        # --- 1. STATE RECONSTRUCTION ---
        # Initialize Left/Right states
        # Left is ALWAYS the Owner (Interior)
        idx_L = g.face_owner
        rho_L, u_L, v_L, p_L, rhoE_L = f.rho[idx_L], f.u[idx_L], f.v[idx_L], f.p[idx_L], f.rhoE[idx_L]
        
        # Initialize Right (Ghost) as a copy of Left (default to Neumann)
        # We will overwrite specific sections below
        rho_R  = rho_L.copy()
        u_R    = u_L.copy()
        v_R    = v_L.copy()
        p_R    = p_L.copy()
        rhoE_R = rhoE_L.copy()
        
        # A. INTERNAL FACES: Neighbor is the Right state
        mask_int = self.mask_internal
        idx_neigh = g.face_neighbor[mask_int]
        rho_R[mask_int]  = f.rho[idx_neigh]
        u_R[mask_int]    = f.u[idx_neigh]
        v_R[mask_int]    = f.v[idx_neigh]
        p_R[mask_int]    = f.p[idx_neigh]
        rhoE_R[mask_int] = f.rhoE[idx_neigh]

        # B. INLET FACES: Force Stagnation State
        # Ghost cell is a high-pressure tank with u=0.
        # This pressure difference drives the flow.
        mask_in = self.mask_inlet
        rho_R[mask_in]  = self.rho_stag
        p_R[mask_in]    = self.P_stag
        u_R[mask_in]    = 0.0
        v_R[mask_in]    = 0.0
        rhoE_R[mask_in] = self.rhoE_stag

        # C. WALL FACES: Reflect Velocity (Slip Wall)
        # Ghost Density/Pressure = Owner Density/Pressure
        # Ghost Velocity = Mirror of Owner Velocity across Normal
        mask_w = self.mask_wall
        
        # Get normals for wall faces
        nx_w = g.face_normals_x[mask_w]
        ny_w = g.face_normals_y[mask_w]
        
        # V_normal = V dot n
        vn = u_L[mask_w] * nx_w + v_L[mask_w] * ny_w
        
        # V_reflected = V - 2 * V_normal * n
        u_R[mask_w] = u_L[mask_w] - 2 * vn * nx_w
        v_R[mask_w] = v_L[mask_w] - 2 * vn * ny_w
        
        # D. OUTLET FACES:
        # Already handled by the "Copy" default (Neumann condition).
        # This lets waves exit the domain.

        # --- 2. RUSANOV FLUX CALCULATION ---
        # (Same math as before, but now using our corrected Ghost States)
        
        nx = g.face_normals_x
        ny = g.face_normals_y
        area = g.face_areas
        
        # Enthalpy
        H_L = (rhoE_L + p_L) / rho_L
        H_R = (rhoE_R + p_R) / rho_R
        
        # Normal Velocity
        un_L = u_L * nx + v_L * ny
        un_R = u_R * nx + v_R * ny
        
        # Flux Vectors F(U)
        # Mass
        flux_rho_L = rho_L * un_L
        flux_rho_R = rho_R * un_R
        # Momentum X
        flux_rhou_L = rho_L * u_L * un_L + p_L * nx
        flux_rhou_R = rho_R * u_R * un_R + p_R * nx
        # Momentum Y
        flux_rhov_L = rho_L * v_L * un_L + p_L * ny
        flux_rhov_R = rho_R * v_R * un_R + p_R * ny
        # Energy
        flux_rhoE_L = rho_L * H_L * un_L
        flux_rhoE_R = rho_R * H_R * un_R
        
        # Dissipation (Wave Speeds)
        c_L = np.sqrt(f.gamma * np.maximum(p_L, 1e-5) / np.maximum(rho_L, 1e-5))
        c_R = np.sqrt(f.gamma * np.maximum(p_R, 1e-5) / np.maximum(rho_R, 1e-5))
        lambda_L = np.abs(un_L) + c_L
        lambda_R = np.abs(un_R) + c_R
        alpha = np.maximum(lambda_L, lambda_R)
        
        # Final Flux
        self.flux_rho  = (0.5 * (flux_rho_L  + flux_rho_R)  - 0.5 * alpha * (rho_R  - rho_L)) * area
        self.flux_rhou = (0.5 * (flux_rhou_L + flux_rhou_R) - 0.5 * alpha * (rho_R * u_R - rho_L * u_L)) * area
        self.flux_rhov = (0.5 * (flux_rhov_L + flux_rhov_R) - 0.5 * alpha * (rho_R * v_R - rho_L * v_L)) * area
        self.flux_rhoE = (0.5 * (flux_rhoE_L + flux_rhoE_R) - 0.5 * alpha * (rhoE_R - rhoE_L)) * area

    def update_field(self, dt):
        g = self.grid
        f = self.field
        
        d_rho  = np.zeros(g.num_cells)
        d_rhou = np.zeros(g.num_cells)
        d_rhov = np.zeros(g.num_cells)
        d_rhoE = np.zeros(g.num_cells)
        
        # Subtract from Owner
        np.add.at(d_rho,  g.face_owner, -self.flux_rho)
        np.add.at(d_rhou, g.face_owner, -self.flux_rhou)
        np.add.at(d_rhov, g.face_owner, -self.flux_rhov)
        np.add.at(d_rhoE, g.face_owner, -self.flux_rhoE)
        
        # Add to Neighbor (Internal Only)
        mask = (g.face_neighbor != -1)
        valid_neigh = g.face_neighbor[mask]
        
        np.add.at(d_rho,  valid_neigh, self.flux_rho[mask])
        np.add.at(d_rhou, valid_neigh, self.flux_rhou[mask])
        np.add.at(d_rhov, valid_neigh, self.flux_rhov[mask])
        np.add.at(d_rhoE, valid_neigh, self.flux_rhoE[mask])
        
        # Update
        factor = dt / g.cell_volumes
        f.rho  += factor * d_rho
        f.rhou += factor * d_rhou
        f.rhov += factor * d_rhov
        f.rhoE += factor * d_rhoE
        
        f.primitives_from_conservatives()