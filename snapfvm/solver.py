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

        # --- BOUNDARY MASKS ---
        tags = np.array(grid.face_tags, dtype=object)
        self.mask_inlet  = (tags == "Left")
        self.mask_outlet = (tags == "Right")
        self.mask_wall   = (tags == "Top") | (tags == "Bottom") | (tags == "Wall") | (tags == "Centerline")
        self.mask_internal = (grid.face_neighbor != -1)
        
        print(f"Solver Init: Found {np.sum(self.mask_inlet)} Inlets, "
              f"{np.sum(self.mask_outlet)} Outlets, {np.sum(self.mask_wall)} Walls")
        
        # Stagnation Conditions
        self.P_stag = 101325.0
        self.T_stag = 300.0
        self.rho_stag = self.P_stag / (287.05 * self.T_stag)
        self.rhoE_stag = self.P_stag / (field.gamma - 1.0) 

        # Gradient Arrays (New for 2nd Order)
        # We store x-gradient and y-gradient for each primitive variable
        N = grid.num_cells
        self.grad_rho_x = np.zeros(N); self.grad_rho_y = np.zeros(N)
        self.grad_u_x   = np.zeros(N); self.grad_u_y   = np.zeros(N)
        self.grad_v_x   = np.zeros(N); self.grad_v_y   = np.zeros(N)
        self.grad_p_x   = np.zeros(N); self.grad_p_y   = np.zeros(N)

    def compute_time_step(self, cfl=0.5):
        # (Same as before)
        g = self.grid
        f = self.field
        c = np.sqrt(f.gamma * np.maximum(f.p, 1e-5) / np.maximum(f.rho, 1e-5))
        vel_mag = np.sqrt(f.u**2 + f.v**2)
        lambda_max = vel_mag + c
        length_scale = np.sqrt(g.cell_volumes)
        local_dt = length_scale / (lambda_max + 1e-12)
        return cfl * np.min(local_dt)

    def compute_gradients(self):
        """
        GREEN-GAUSS GRADIENT CALCULATION
        Computes gradients for rho, u, v, p at every cell center.
        """
        g = self.grid
        f = self.field
        
        # Reset gradients
        self.grad_rho_x.fill(0); self.grad_rho_y.fill(0)
        self.grad_u_x.fill(0);   self.grad_u_y.fill(0)
        self.grad_v_x.fill(0);   self.grad_v_y.fill(0)
        self.grad_p_x.fill(0);   self.grad_p_y.fill(0)
        
        # --- 1. Face Values (Arithmetic Average) ---
        # Left State
        idx_L = g.face_owner
        
        # Right State (Ghost construction needed for boundaries)
        idx_neigh = g.face_neighbor
        
        # Internal Neighbors
        mask_int = self.mask_internal
        valid_neigh = idx_neigh[mask_int]
        
        # Helper to get full Right State array
        # Default to Owner value (Zero Gradient) for boundaries
        rho_R = f.rho[idx_L].copy(); u_R = f.u[idx_L].copy()
        v_R   = f.v[idx_L].copy();   p_R = f.p[idx_L].copy()
        
        # Fill Internal
        rho_R[mask_int] = f.rho[valid_neigh]
        u_R[mask_int]   = f.u[valid_neigh]
        v_R[mask_int]   = f.v[valid_neigh]
        p_R[mask_int]   = f.p[valid_neigh]
        
        # Fill Inlet (Fixed Value)
        mask_in = self.mask_inlet
        rho_R[mask_in] = self.rho_stag
        p_R[mask_in]   = self.P_stag
        u_R[mask_in]   = 0.0
        v_R[mask_in]   = 0.0
        
        # Fill Wall (Slip - Mirror Velocity) -- Optional for gradients, 
        # but simpler to just use Zero Gradient (copy) for robustness initially.
        # We leave Wall/Outlet as Copy (Neumann) for the gradient calc.

        # --- 2. Face Center Average ---
        rho_f = 0.5 * (f.rho[idx_L] + rho_R)
        u_f   = 0.5 * (f.u[idx_L]   + u_R)
        v_f   = 0.5 * (f.v[idx_L]   + v_R)
        p_f   = 0.5 * (f.p[idx_L]   + p_R)

        # --- 3. Green-Gauss Integration ---
        # Gradient += Value_Face * Normal * Area
        # Note: Normals point OUT of Owner, INTO Neighbor
        
        nx = g.face_normals_x
        ny = g.face_normals_y
        area = g.face_areas
        
        # Weighted Normals
        ax = nx * area
        ay = ny * area
        
        # Accumulate to Owner
        np.add.at(self.grad_rho_x, idx_L, rho_f * ax)
        np.add.at(self.grad_rho_y, idx_L, rho_f * ay)
        np.add.at(self.grad_u_x,   idx_L, u_f   * ax)
        np.add.at(self.grad_u_y,   idx_L, u_f   * ay)
        np.add.at(self.grad_v_x,   idx_L, v_f   * ax)
        np.add.at(self.grad_v_y,   idx_L, v_f   * ay)
        np.add.at(self.grad_p_x,   idx_L, p_f   * ax)
        np.add.at(self.grad_p_y,   idx_L, p_f   * ay)
        
        # Accumulate to Neighbor (Subtract because normal points IN)
        np.add.at(self.grad_rho_x, valid_neigh, -rho_f[mask_int] * ax[mask_int])
        np.add.at(self.grad_rho_y, valid_neigh, -rho_f[mask_int] * ay[mask_int])
        np.add.at(self.grad_u_x,   valid_neigh, -u_f[mask_int]   * ax[mask_int])
        np.add.at(self.grad_u_y,   valid_neigh, -u_f[mask_int]   * ay[mask_int])
        np.add.at(self.grad_v_x,   valid_neigh, -v_f[mask_int]   * ax[mask_int])
        np.add.at(self.grad_v_y,   valid_neigh, -v_f[mask_int]   * ay[mask_int])
        np.add.at(self.grad_p_x,   valid_neigh, -p_f[mask_int]   * ax[mask_int])
        np.add.at(self.grad_p_y,   valid_neigh, -p_f[mask_int]   * ay[mask_int])

        # --- 4. Divide by Volume ---
        vol = g.cell_volumes
        self.grad_rho_x /= vol; self.grad_rho_y /= vol
        self.grad_u_x   /= vol; self.grad_u_y   /= vol
        self.grad_v_x   /= vol; self.grad_v_y   /= vol
        self.grad_p_x   /= vol; self.grad_p_y   /= vol


    def compute_fluxes(self):
        g = self.grid
        f = self.field
        
        # 1. COMPUTE GRADIENTS
        self.compute_gradients()
        
        # 2. RECONSTRUCTION VECTORS
        # Vector from Cell Center -> Face Center
        idx_L = g.face_owner
        dx_L = g.face_centers_x - g.cell_centers_x[idx_L]
        dy_L = g.face_centers_y - g.cell_centers_y[idx_L]
        
        # Reconstruct Left State (Linear Extrapolation)
        rho_L = f.rho[idx_L] + (self.grad_rho_x[idx_L]*dx_L + self.grad_rho_y[idx_L]*dy_L)
        u_L   = f.u[idx_L]   + (self.grad_u_x[idx_L]*dx_L   + self.grad_u_y[idx_L]*dy_L)
        v_L   = f.v[idx_L]   + (self.grad_v_x[idx_L]*dx_L   + self.grad_v_y[idx_L]*dy_L)
        p_L   = f.p[idx_L]   + (self.grad_p_x[idx_L]*dx_L   + self.grad_p_y[idx_L]*dy_L)
        
        # --- SAFETY CLAMP (CRITICAL FOR STABILITY) ---
        # Prevent gradients from creating negative physics
        rho_L = np.maximum(rho_L, 1e-6)
        p_L   = np.maximum(p_L, 1e-6)

        # For Right State, we start with Neighbors
        rho_R = f.rho[idx_L].copy(); u_R = f.u[idx_L].copy()
        v_R   = f.v[idx_L].copy();   p_R = f.p[idx_L].copy()
        
        # Internal Neighbors: Reconstruct from Neighbor Center
        mask_int = self.mask_internal
        idx_neigh = g.face_neighbor[mask_int]
        
        dx_R = g.face_centers_x[mask_int] - g.cell_centers_x[idx_neigh]
        dy_R = g.face_centers_y[mask_int] - g.cell_centers_y[idx_neigh]
        
        rho_R[mask_int] = f.rho[idx_neigh] + (self.grad_rho_x[idx_neigh]*dx_R + self.grad_rho_y[idx_neigh]*dy_R)
        u_R[mask_int]   = f.u[idx_neigh]   + (self.grad_u_x[idx_neigh]*dx_R   + self.grad_u_y[idx_neigh]*dy_R)
        v_R[mask_int]   = f.v[idx_neigh]   + (self.grad_v_x[idx_neigh]*dx_R   + self.grad_v_y[idx_neigh]*dy_R)
        p_R[mask_int]   = f.p[idx_neigh]   + (self.grad_p_x[idx_neigh]*dx_R   + self.grad_p_y[idx_neigh]*dy_R)

        # --- SAFETY CLAMP FOR RIGHT STATE ---
        rho_R = np.maximum(rho_R, 1e-6)
        p_R   = np.maximum(p_R, 1e-6)

        # 3. APPLY BOUNDARY CONDITIONS (Ghost Values)
        
        # Inlet (Hard Set)
        mask_in = self.mask_inlet
        rho_R[mask_in] = self.rho_stag
        p_R[mask_in]   = self.P_stag
        u_R[mask_in]   = 0.0
        v_R[mask_in]   = 0.0
        # Check inlet values are safe (redundant but good practice)
        rho_R[mask_in] = np.maximum(rho_R[mask_in], 1e-6)
        p_R[mask_in]   = np.maximum(p_R[mask_in], 1e-6)
        
        # Wall (Slip)
        mask_w = self.mask_wall
        nx_w = g.face_normals_x[mask_w]
        ny_w = g.face_normals_y[mask_w]
        
        # Re-calc normal velocity using the SAFE Left state
        vn = u_L[mask_w] * nx_w + v_L[mask_w] * ny_w
        
        u_R[mask_w] = u_L[mask_w] - 2 * vn * nx_w
        v_R[mask_w] = v_L[mask_w] - 2 * vn * ny_w
        rho_R[mask_w] = rho_L[mask_w]
        p_R[mask_w]   = p_L[mask_w]

        # 4. RECOMPUTE CONSERVATIVES (rhoE) FROM PRIMITIVES
        # Using the now-safe Pressure and Density
        rhoE_L = (p_L / (f.gamma - 1.0)) + 0.5 * rho_L * (u_L**2 + v_L**2)
        rhoE_R = (p_R / (f.gamma - 1.0)) + 0.5 * rho_R * (u_R**2 + v_R**2)

        # 5. RUSANOV FLUX (Identical Logic to before)
        nx = g.face_normals_x
        ny = g.face_normals_y
        area = g.face_areas
        
        H_L = (rhoE_L + p_L) / rho_L
        H_R = (rhoE_R + p_R) / rho_R
        
        un_L = u_L * nx + v_L * ny
        un_R = u_R * nx + v_R * ny
        
        flux_rho_L = rho_L * un_L
        flux_rho_R = rho_R * un_R
        
        flux_rhou_L = rho_L * u_L * un_L + p_L * nx
        flux_rhou_R = rho_R * u_R * un_R + p_R * nx
        
        flux_rhov_L = rho_L * v_L * un_L + p_L * ny
        flux_rhov_R = rho_R * v_R * un_R + p_R * ny
        
        flux_rhoE_L = rho_L * H_L * un_L
        flux_rhoE_R = rho_R * H_R * un_R
        
        # Sound speed check (redundant but prevents sqrt crash)
        p_L = np.maximum(p_L, 1e-6); rho_L = np.maximum(rho_L, 1e-6)
        p_R = np.maximum(p_R, 1e-6); rho_R = np.maximum(rho_R, 1e-6)
        
        c_L = np.sqrt(f.gamma * p_L / rho_L)
        c_R = np.sqrt(f.gamma * p_R / rho_R)
        lambda_L = np.abs(un_L) + c_L
        lambda_R = np.abs(un_R) + c_R
        alpha = np.maximum(lambda_L, lambda_R)
        
        self.flux_rho  = (0.5 * (flux_rho_L  + flux_rho_R)  - 0.5 * alpha * (rho_R  - rho_L)) * area
        self.flux_rhou = (0.5 * (flux_rhou_L + flux_rhou_R) - 0.5 * alpha * (rho_R * u_R - rho_L * u_L)) * area
        self.flux_rhov = (0.5 * (flux_rhov_L + flux_rhov_R) - 0.5 * alpha * (rho_R * v_R - rho_L * v_L)) * area
        self.flux_rhoE = (0.5 * (flux_rhoE_L + flux_rhoE_R) - 0.5 * alpha * (rhoE_R - rhoE_L)) * area
        
        
        
    def update_field(self, dt):
        # (Same as before)
        g = self.grid
        f = self.field
        
        d_rho  = np.zeros(g.num_cells)
        d_rhou = np.zeros(g.num_cells)
        d_rhov = np.zeros(g.num_cells)
        d_rhoE = np.zeros(g.num_cells)
        
        np.add.at(d_rho,  g.face_owner, -self.flux_rho)
        np.add.at(d_rhou, g.face_owner, -self.flux_rhou)
        np.add.at(d_rhov, g.face_owner, -self.flux_rhov)
        np.add.at(d_rhoE, g.face_owner, -self.flux_rhoE)
        
        mask = (g.face_neighbor != -1)
        valid_neigh = g.face_neighbor[mask]
        
        np.add.at(d_rho,  valid_neigh, self.flux_rho[mask])
        np.add.at(d_rhou, valid_neigh, self.flux_rhou[mask])
        np.add.at(d_rhov, valid_neigh, self.flux_rhov[mask])
        np.add.at(d_rhoE, valid_neigh, self.flux_rhoE[mask])
        
        factor = dt / g.cell_volumes
        f.rho  += factor * d_rho
        f.rhou += factor * d_rhou
        f.rhov += factor * d_rhov
        f.rhoE += factor * d_rhoE
        
        f.primitives_from_conservatives()