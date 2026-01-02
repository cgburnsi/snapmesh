"""
snapfvm/solver.py
-----------------
Generic FVM Solver.
UPDATED: Uses Face-Flux logic for BCs.
"""
import numpy as np

class FiniteVolumeSolver:
    def __init__(self, grid, physics_model):
        self.grid = grid
        self.model = physics_model
        
        self.q = np.zeros((grid.n_cells, self.model.num_variables), dtype=float)
        self.grad_q = np.zeros((grid.n_cells, self.model.num_variables, 2), dtype=float)
        
        self.time = 0.0
        self.iter = 0
        self.total_mass_history = []
        
        print(f"--- FVM Solver Initialized ---")

    def set_initial_condition(self, q_init):
        self.q[:] = q_init
        mass = np.sum(self.q[:, 0] * self.grid.cell_volumes)
        self.total_mass_history.append(mass)

    def compute_gradients(self):
        # (Standard Green-Gauss)
        self.grad_q.fill(0.0)
        for i in range(self.grid.n_faces):
            c_left = self.grid.face_cells[i, 0]
            c_right = self.grid.face_cells[i, 1]
            normal = self.grid.face_normals[i]
            
            q_L = self.q[c_left]
            q_R = self.q[c_right] if c_right != -1 else q_L
                
            q_face = 0.5 * (q_L + q_R)
            term = np.outer(q_face, normal)
            self.grad_q[c_left] += term
            if c_right != -1: self.grad_q[c_right] -= term
                
        for c in range(self.grid.n_cells):
            self.grad_q[c] /= self.grid.cell_volumes[c]

    def step(self, dt):
        n_vars = self.model.num_variables
        residuals = np.zeros((self.grid.n_cells, n_vars), dtype=float)
        net_boundary_flux = np.zeros(n_vars, dtype=float)

        if self.model.has_viscous_terms:
            self.compute_gradients()

        for i in range(self.grid.n_faces):
            c_left = self.grid.face_cells[i, 0]
            c_right = self.grid.face_cells[i, 1]
            normal = self.grid.face_normals[i]

            q_L = self.q[c_left]
            
            if c_right == -1:
                # --- BOUNDARY FLUX ---
                group_id = self.grid.face_groups[i]
                bc_name = self.grid.group_names.get(group_id, "unknown")
                
                flux = self.model.compute_boundary_flux(q_L, normal, bc_name)
                net_boundary_flux += flux # Accumulate for audit
            else:
                # --- INTERNAL FLUX ---
                q_R = self.q[c_right]
                flux = self.model.compute_flux(q_L, q_R, normal)
            
            # Update Residuals
            residuals[c_left] -= flux
            if c_right != -1:
                residuals[c_right] += flux

        # Time Integration
        d_q = (dt / self.grid.cell_volumes[:, None]) * residuals
        self.q += d_q
        
        # Audit
        current_mass = np.sum(self.q[:, 0] * self.grid.cell_volumes)
        prev_mass = self.total_mass_history[-1]
        mass_change = current_mass - prev_mass
        boundary_loss = np.sum(net_boundary_flux[0]) * dt
        balance_error = mass_change + boundary_loss
        
        if abs(balance_error) > 1e-10:
             print(f"[!] AUDIT WARNING: Mass Imbalance = {balance_error:.2e}")
        
        self.total_mass_history.append(current_mass)
        self.time += dt
        self.iter += 1
        
        return np.max(np.abs(residuals))



















































'''  The following is the original solver code.  It is for reference only now.


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

        # Gradient Arrays
        N = grid.num_cells
        self.grad_rho_x = np.zeros(N); self.grad_rho_y = np.zeros(N)
        self.grad_u_x   = np.zeros(N); self.grad_u_y   = np.zeros(N)
        self.grad_v_x   = np.zeros(N); self.grad_v_y   = np.zeros(N)
        self.grad_p_x   = np.zeros(N); self.grad_p_y   = np.zeros(N)

    def compute_time_step(self, cfl=0.5):
        g = self.grid
        f = self.field
        # Safe sound speed
        c = np.sqrt(f.gamma * np.maximum(f.p, 1e-5) / np.maximum(f.rho, 1e-5))
        vel_mag = np.sqrt(f.u**2 + f.v**2)
        local_dt = np.sqrt(g.cell_volumes) / (vel_mag + c + 1e-12)
        return cfl * np.min(local_dt)

    def compute_gradients(self):
        """Green-Gauss Gradient Calculation with Correct Wall Reflection"""
        g = self.grid
        f = self.field
        
        # Reset
        self.grad_rho_x.fill(0); self.grad_rho_y.fill(0)
        self.grad_u_x.fill(0);   self.grad_u_y.fill(0)
        self.grad_v_x.fill(0);   self.grad_v_y.fill(0)
        self.grad_p_x.fill(0);   self.grad_p_y.fill(0)
        
        # 1. Reconstruct Face Values
        idx_L = g.face_owner
        
        rho_R = f.rho[idx_L].copy(); u_R = f.u[idx_L].copy()
        v_R   = f.v[idx_L].copy();   p_R = f.p[idx_L].copy()
        
        # Internal Neighbors
        mask_int = self.mask_internal
        idx_neigh = g.face_neighbor[mask_int]
        rho_R[mask_int] = f.rho[idx_neigh]; u_R[mask_int] = f.u[idx_neigh]
        v_R[mask_int] = f.v[idx_neigh];     p_R[mask_int] = f.p[idx_neigh]
        
        # Inlet Fixed
        mask_in = self.mask_inlet
        rho_R[mask_in] = self.rho_stag; p_R[mask_in] = self.P_stag
        u_R[mask_in] = 0.0;             v_R[mask_in] = 0.0

        # Wall (Reflected Velocity) - CRITICAL for accurate wall gradients
        mask_w = self.mask_wall
        nx_w, ny_w = g.face_normals_x[mask_w], g.face_normals_y[mask_w]
        
        # V_reflected = V - 2(V.n)n
        vn = u_R[mask_w] * nx_w + v_R[mask_w] * ny_w
        u_R[mask_w] = u_R[mask_w] - 2 * vn * nx_w
        v_R[mask_w] = v_R[mask_w] - 2 * vn * ny_w
        # rho/p remain copies (Neumann)

        # Face Averages
        rho_f = 0.5 * (f.rho[idx_L] + rho_R)
        u_f   = 0.5 * (f.u[idx_L] + u_R)
        v_f   = 0.5 * (f.v[idx_L] + v_R)
        p_f   = 0.5 * (f.p[idx_L] + p_R)
        
        # Accumulate
        nx, ny, area = g.face_normals_x, g.face_normals_y, g.face_areas
        ax, ay = nx * area, ny * area
        
        # Add to Owner
        np.add.at(self.grad_rho_x, idx_L, rho_f * ax)
        np.add.at(self.grad_rho_y, idx_L, rho_f * ay)
        np.add.at(self.grad_u_x,   idx_L, u_f * ax)
        np.add.at(self.grad_u_y,   idx_L, u_f * ay)
        np.add.at(self.grad_v_x,   idx_L, v_f * ax)
        np.add.at(self.grad_v_y,   idx_L, v_f * ay)
        np.add.at(self.grad_p_x,   idx_L, p_f * ax)
        np.add.at(self.grad_p_y,   idx_L, p_f * ay)
        
        # Subtract from Neighbor
        np.add.at(self.grad_rho_x, idx_neigh, -rho_f[mask_int] * ax[mask_int])
        np.add.at(self.grad_rho_y, idx_neigh, -rho_f[mask_int] * ay[mask_int])
        np.add.at(self.grad_u_x,   idx_neigh, -u_f[mask_int] * ax[mask_int])
        np.add.at(self.grad_u_y,   idx_neigh, -u_f[mask_int] * ay[mask_int])
        np.add.at(self.grad_v_x,   idx_neigh, -v_f[mask_int] * ax[mask_int])
        np.add.at(self.grad_v_y,   idx_neigh, -v_f[mask_int] * ay[mask_int])
        np.add.at(self.grad_p_x,   idx_neigh, -p_f[mask_int] * ax[mask_int])
        np.add.at(self.grad_p_y,   idx_neigh, -p_f[mask_int] * ay[mask_int])
        
        # Divide by Vol
        vol = g.cell_volumes
        self.grad_rho_x /= vol; self.grad_rho_y /= vol
        self.grad_u_x /= vol;   self.grad_u_y /= vol
        self.grad_v_x /= vol;   self.grad_v_y /= vol
        self.grad_p_x /= vol;   self.grad_p_y /= vol

    def limit_gradients(self):
        """
        VENKATAKRISHNAN LIMITER
        Smoother than Barth-Jespersen, allows expansion peaks to survive.
        """
        g = self.grid
        f = self.field
        
        # 1. Find Min/Max
        rho_min = f.rho.copy(); rho_max = f.rho.copy()
        p_min   = f.p.copy();   p_max   = f.p.copy()
        
        idx_L = g.face_owner
        mask_int = self.mask_internal
        idx_R = g.face_neighbor[mask_int]
        
        np.minimum.at(rho_min, idx_L[mask_int], f.rho[idx_R])
        np.maximum.at(rho_max, idx_L[mask_int], f.rho[idx_R])
        np.minimum.at(p_min,   idx_L[mask_int], f.p[idx_R])
        np.maximum.at(p_max,   idx_L[mask_int], f.p[idx_R])
        
        np.minimum.at(rho_min, idx_R, f.rho[idx_L[mask_int]])
        np.maximum.at(rho_max, idx_R, f.rho[idx_L[mask_int]])
        np.minimum.at(p_min,   idx_R, f.p[idx_L[mask_int]])
        np.maximum.at(p_max,   idx_R, f.p[idx_L[mask_int]])

        # Venkatakrishnan Parameter (K)
        # K determines how much wiggle room we allow.
        # K=5 is standard for subsonic/supersonic flows.
        K = 5.0
        
        # Compute Cell Volume Scale (Delta_x^3 roughly, or just Volume)
        # We need a characteristic delta for the limiter threshold
        vol = g.cell_volumes
        delta_mesh = np.sqrt(vol) # Char length
        epsilon_sq = (K * delta_mesh)**3 # Threshold parameter
        
        # --- Helper for Venkat Logic ---
        def venkat_phi(val_c, val_min, val_max, grad_x, grad_y, dx, dy):
            # Compute projection to face
            delta2 = grad_x*dx + grad_y*dy
            
            # Since we need phi for the *whole cell*, we really want to check
            # the max projection to ALL faces and limit based on that.
            # But the standard implementation computes a phi for each face
            # and takes the minimum phi.
            
            # This is hard to vectorize perfectly without a loop over faces-of-cell.
            # Instead, we use the Barth-Jespersen structure but with the Venkat function.
            
            # Approximate: Treat each face independently
            return delta2 # Placeholder for structure below
            
        # Implementing full Venkat vectorized is tricky.
        # Let's use a "Softened Barth-Jespersen" which is 90% of Venkat's benefit.
        # alpha = min(1, num/den)
        # num = (d_max^2 + eps^2) + 2*delta*d_max
        # den = (d_max^2 + 2*delta^2 + d_max*delta + eps^2)
        
        # To keep this code robust and runnable right now without complex loops:
        # We will simply RELAX the Barth-Jespersen limiter by a factor.
        # This is a common engineering hack.
        
        # Initialize Alpha
        alpha_rho = np.ones(g.num_cells)
        alpha_p   = np.ones(g.num_cells)
        
        dx = g.face_centers_x - g.cell_centers_x[idx_L]
        dy = g.face_centers_y - g.cell_centers_y[idx_L]
        
        delta_rho = self.grad_rho_x[idx_L] * dx + self.grad_rho_y[idx_L] * dy
        delta_p   = self.grad_p_x[idx_L] * dx   + self.grad_p_y[idx_L] * dy
        
        def soft_bj(d_phi, phi_c, phi_min, phi_max):
            # Safe divide
            d_phi = np.where(np.abs(d_phi) < 1e-12, 1e-12, d_phi)
            
            r = np.ones_like(d_phi)
            mask_pos = d_phi > 0
            mask_neg = d_phi < 0
            
            # Standard BJ Ratio
            r[mask_pos] = (phi_max - phi_c)[mask_pos] / d_phi[mask_pos]
            r[mask_neg] = (phi_min - phi_c)[mask_neg] / d_phi[mask_neg]
            
            # SOFTENER: Instead of min(1, r), we use a smooth function
            # Venkat-like smoother: phi(r) = (r^2 + 2r) / (r^2 + r + 2)
            # This assumes r > 0. If r > 1, it approaches 1.
            # But r here is the ratio limit/delta.
            
            # Simple Relaxation: Allow it to overshoot by 20%? No, monotonicity.
            
            # Let's implement the actual Venkat function for the limiter 'phi'
            # phi = ((dist_to_max)^2 + eps) ...
            
            # FALLBACK: Just use BJ but clamped less aggressively? 
            # No, let's use the actual Venkat formula.
            
            # Re-map: 
            # D_max = phi_max - phi_c
            # Delta = d_phi
            # If Delta > 0:
            #   phi = (D_max^2 + eps^2 + 2*Delta*D_max) / (D_max^2 + 2*Delta^2 + D_max*Delta + eps^2)
            
            D_max = np.zeros_like(d_phi)
            D_max[mask_pos] = (phi_max - phi_c)[mask_pos]
            D_max[mask_neg] = (phi_min - phi_c)[mask_neg]
            
            # Use a fixed epsilon for the whole domain
            eps2 = (1e-4 * np.max(phi_c))**2 # Scale by field magnitude
            
            Delta = d_phi
            
            num = D_max**2 + eps2 + 2*Delta*D_max
            den = D_max**2 + 2*Delta**2 + D_max*Delta + eps2
            
            # If Delta is opposite sign to D_max (shouldn't happen with min/max bounds), handle it
            # Actually Venkat assumes you compare to the correct bound.
            
            phi = num / den
            return phi

        # Calculate Smooth Alphas
        a_rho_f = soft_bj(delta_rho, f.rho[idx_L], rho_min[idx_L], rho_max[idx_L])
        a_p_f   = soft_bj(delta_p,   f.p[idx_L],   p_min[idx_L],   p_max[idx_L])
        
        np.minimum.at(alpha_rho, idx_L, a_rho_f)
        np.minimum.at(alpha_p,   idx_L, a_p_f)
        
        # Repeat for Neighbors
        dx_R = g.face_centers_x[mask_int] - g.cell_centers_x[idx_R]
        dy_R = g.face_centers_y[mask_int] - g.cell_centers_y[idx_R]
        
        d_rho_R = self.grad_rho_x[idx_R]*dx_R + self.grad_rho_y[idx_R]*dy_R
        d_p_R   = self.grad_p_x[idx_R]*dx_R   + self.grad_p_y[idx_R]*dy_R
        
        a_rho_R = soft_bj(d_rho_R, f.rho[idx_R], rho_min[idx_R], rho_max[idx_R])
        a_p_R   = soft_bj(d_p_R,   f.p[idx_R],   p_min[idx_R],   p_max[idx_R])
        
        np.minimum.at(alpha_rho, idx_R, a_rho_R)
        np.minimum.at(alpha_p,   idx_R, a_p_R)
        
        # Apply
        self.grad_rho_x *= alpha_rho; self.grad_rho_y *= alpha_rho
        self.grad_p_x   *= alpha_p;   self.grad_p_y   *= alpha_p
        self.grad_u_x   *= alpha_p;   self.grad_u_y   *= alpha_p
        self.grad_v_x   *= alpha_p;   self.grad_v_y   *= alpha_p

    def compute_fluxes(self):
        """HLLC FLUX SCHEME"""
        g = self.grid
        f = self.field
        
        # 1. Gradients & Limiting
        self.compute_gradients()
        self.limit_gradients()
        
        # 2. Reconstruction
        # --- LEFT ---
        idx_L = g.face_owner
        dx_L = g.face_centers_x - g.cell_centers_x[idx_L]
        dy_L = g.face_centers_y - g.cell_centers_y[idx_L]
        
        rho_L = f.rho[idx_L] + (self.grad_rho_x[idx_L]*dx_L + self.grad_rho_y[idx_L]*dy_L)
        u_L   = f.u[idx_L]   + (self.grad_u_x[idx_L]*dx_L   + self.grad_u_y[idx_L]*dy_L)
        v_L   = f.v[idx_L]   + (self.grad_v_x[idx_L]*dx_L   + self.grad_v_y[idx_L]*dy_L)
        p_L   = f.p[idx_L]   + (self.grad_p_x[idx_L]*dx_L   + self.grad_p_y[idx_L]*dy_L)
        
        # --- RIGHT ---
        rho_R = f.rho[idx_L].copy(); u_R = f.u[idx_L].copy()
        v_R = f.v[idx_L].copy();     p_R = f.p[idx_L].copy()
        
        mask_int = self.mask_internal
        idx_neigh = g.face_neighbor[mask_int]
        dx_R = g.face_centers_x[mask_int] - g.cell_centers_x[idx_neigh]
        dy_R = g.face_centers_y[mask_int] - g.cell_centers_y[idx_neigh]
        
        rho_R[mask_int] = f.rho[idx_neigh] + (self.grad_rho_x[idx_neigh]*dx_R + self.grad_rho_y[idx_neigh]*dy_R)
        u_R[mask_int]   = f.u[idx_neigh]   + (self.grad_u_x[idx_neigh]*dx_R   + self.grad_u_y[idx_neigh]*dy_R)
        v_R[mask_int]   = f.v[idx_neigh]   + (self.grad_v_x[idx_neigh]*dx_R   + self.grad_v_y[idx_neigh]*dy_R)
        p_R[mask_int]   = f.p[idx_neigh]   + (self.grad_p_x[idx_neigh]*dx_R   + self.grad_p_y[idx_neigh]*dy_R)

        # --- BOUNDARIES ---
        mask_in = self.mask_inlet
        rho_R[mask_in] = self.rho_stag; p_R[mask_in] = self.P_stag
        u_R[mask_in] = 0.0;             v_R[mask_in] = 0.0
        
        mask_w = self.mask_wall
        nx_w, ny_w = g.face_normals_x[mask_w], g.face_normals_y[mask_w]
        vn_w = u_L[mask_w] * nx_w + v_L[mask_w] * ny_w
        u_R[mask_w] = u_L[mask_w] - 2 * vn_w * nx_w
        v_R[mask_w] = v_L[mask_w] - 2 * vn_w * ny_w
        rho_R[mask_w] = rho_L[mask_w]; p_R[mask_w] = p_L[mask_w]
        
        # Safety Clamps
        rho_L = np.maximum(rho_L, 1e-6); p_L = np.maximum(p_L, 1e-6)
        rho_R = np.maximum(rho_R, 1e-6); p_R = np.maximum(p_R, 1e-6)

        # 3. HLLC FLUX
        nx, ny = g.face_normals_x, g.face_normals_y
        area = g.face_areas
        
        un_L = u_L * nx + v_L * ny
        un_R = u_R * nx + v_R * ny
        
        c_L = np.sqrt(f.gamma * p_L / rho_L)
        c_R = np.sqrt(f.gamma * p_R / rho_R)
        
        S_L = np.minimum(un_L - c_L, un_R - c_R)
        S_R = np.maximum(un_L + c_L, un_R + c_R)
        
        term1 = p_R - p_L + rho_L * un_L * (S_L - un_L) - rho_R * un_R * (S_R - un_R)
        term2 = rho_L * (S_L - un_L) - rho_R * (S_R - un_R)
        S_star = term1 / (term2 + 1e-15)
        
        E_L = (p_L / (f.gamma - 1.0)) + 0.5 * rho_L * (u_L**2 + v_L**2)
        E_R = (p_R / (f.gamma - 1.0)) + 0.5 * rho_R * (u_R**2 + v_R**2)
        H_L = (E_L + p_L) / rho_L
        H_R = (E_R + p_R) / rho_R
        
        F_rho_L = rho_L * un_L
        F_rho_R = rho_R * un_R
        F_rhou_L = rho_L * u_L * un_L + p_L * nx
        F_rhou_R = rho_R * u_R * un_R + p_R * nx
        F_rhov_L = rho_L * v_L * un_L + p_L * ny
        F_rhov_R = rho_R * v_R * un_R + p_R * ny
        F_rhoE_L = rho_L * H_L * un_L
        F_rhoE_R = rho_R * H_R * un_R
        
        self.flux_rho  = F_rho_L.copy()
        self.flux_rhou = F_rhou_L.copy()
        self.flux_rhov = F_rhov_L.copy()
        self.flux_rhoE = F_rhoE_L.copy()
        
        mask_star_L = (S_L <= 0) & (S_star >= 0)
        mask_star_R = (S_star < 0) & (S_R >= 0)
        mask_R      = (S_R < 0)
        
        if np.any(mask_star_L):
            fac_L = rho_L[mask_star_L] * (S_L[mask_star_L] - un_L[mask_star_L]) / (S_L[mask_star_L] - S_star[mask_star_L] + 1e-15)
            dU_rho = fac_L - rho_L[mask_star_L]
            u_star = u_L[mask_star_L] + (S_star[mask_star_L] - un_L[mask_star_L]) * nx[mask_star_L]
            v_star = v_L[mask_star_L] + (S_star[mask_star_L] - un_L[mask_star_L]) * ny[mask_star_L]
            dU_rhou = (fac_L * u_star) - (rho_L[mask_star_L] * u_L[mask_star_L])
            dU_rhov = (fac_L * v_star) - (rho_L[mask_star_L] * v_L[mask_star_L])
            term_e = (S_star[mask_star_L] - un_L[mask_star_L]) * (S_star[mask_star_L] + p_L[mask_star_L]/(rho_L[mask_star_L]*(S_L[mask_star_L]-un_L[mask_star_L])+1e-15))
            E_star = fac_L * ((E_L[mask_star_L] / rho_L[mask_star_L]) + term_e)
            dU_rhoE = E_star - E_L[mask_star_L]
            
            SL_ = S_L[mask_star_L]
            self.flux_rho[mask_star_L]  += SL_ * dU_rho
            self.flux_rhou[mask_star_L] += SL_ * dU_rhou
            self.flux_rhov[mask_star_L] += SL_ * dU_rhov
            self.flux_rhoE[mask_star_L] += SL_ * dU_rhoE

        if np.any(mask_star_R):
            fac_R = rho_R[mask_star_R] * (S_R[mask_star_R] - un_R[mask_star_R]) / (S_R[mask_star_R] - S_star[mask_star_R] + 1e-15)
            dU_rho = fac_R - rho_R[mask_star_R]
            u_star = u_R[mask_star_R] + (S_star[mask_star_R] - un_R[mask_star_R]) * nx[mask_star_R]
            v_star = v_R[mask_star_R] + (S_star[mask_star_R] - un_R[mask_star_R]) * ny[mask_star_R]
            dU_rhou = (fac_R * u_star) - (rho_R[mask_star_R] * u_R[mask_star_R])
            dU_rhov = (fac_R * v_star) - (rho_R[mask_star_R] * v_R[mask_star_R])
            term_e = (S_star[mask_star_R] - un_R[mask_star_R]) * (S_star[mask_star_R] + p_R[mask_star_R]/(rho_R[mask_star_R]*(S_R[mask_star_R]-un_R[mask_star_R])+1e-15))
            E_star = fac_R * ((E_R[mask_star_R] / rho_R[mask_star_R]) + term_e)
            dU_rhoE = E_star - E_R[mask_star_R]
            
            SR_ = S_R[mask_star_R]
            # Must overwrite with R flux first, then add correction (Standard HLLC formula)
            self.flux_rho[mask_star_R]  = F_rho_R[mask_star_R]  + SR_ * dU_rho
            self.flux_rhou[mask_star_R] = F_rhou_R[mask_star_R] + SR_ * dU_rhou
            self.flux_rhov[mask_star_R] = F_rhov_R[mask_star_R] + SR_ * dU_rhov
            self.flux_rhoE[mask_star_R] = F_rhoE_R[mask_star_R] + SR_ * dU_rhoE

        if np.any(mask_R):
            self.flux_rho[mask_R]  = F_rho_R[mask_R]
            self.flux_rhou[mask_R] = F_rhou_R[mask_R]
            self.flux_rhov[mask_R] = F_rhov_R[mask_R]
            self.flux_rhoE[mask_R] = F_rhoE_R[mask_R]
            
        self.flux_rho *= area; self.flux_rhou *= area
        self.flux_rhov *= area; self.flux_rhoE *= area

    def update_field(self, dt):
        """
        Applies fluxes AND adds the Axisymmetric Source Term.
        """
        g = self.grid
        f = self.field
        
        d_rho  = np.zeros(g.num_cells)
        d_rhou = np.zeros(g.num_cells)
        d_rhov = np.zeros(g.num_cells)
        d_rhoE = np.zeros(g.num_cells)
        
        # 1. Fluxes (Same as before)
        np.add.at(d_rho,  g.face_owner, -self.flux_rho)
        np.add.at(d_rhou, g.face_owner, -self.flux_rhou)
        np.add.at(d_rhov, g.face_owner, -self.flux_rhov)
        np.add.at(d_rhoE, g.face_owner, -self.flux_rhoE)
        
        mask = (g.face_neighbor != -1)
        valid = g.face_neighbor[mask]
        np.add.at(d_rho,  valid, self.flux_rho[mask])
        np.add.at(d_rhou, valid, self.flux_rhou[mask])
        np.add.at(d_rhov, valid, self.flux_rhov[mask])
        np.add.at(d_rhoE, valid, self.flux_rhoE[mask])
        
        # --- 2. AXISYMMETRIC SOURCE TERM (Hoop Stress) ---
        # Equation: S_y = Pressure / Radius
        # We add (S * Volume) to the momentum accumulator
        
        radius = g.cell_centers_y
        # Avoid divide by zero at centerline
        radius = np.maximum(radius, 1e-6)
        
        # Source = (P / r) * Volume
        # This force acts in the +y (Radial) direction
        hoop_force = (f.p / radius) * g.cell_volumes
        
        # Add to Radial Momentum (rhov)
        d_rhov += hoop_force
        
        # -------------------------------------------------

        # 3. Apply Update
        factor = dt / g.cell_volumes
        f.rho  += factor * d_rho
        f.rhou += factor * d_rhou
        f.rhov += factor * d_rhov
        f.rhoE += factor * d_rhoE
        
        f.primitives_from_conservatives()
        
        
        
        
        
        
        
        
        
        
'''