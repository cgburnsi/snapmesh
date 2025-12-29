import numpy as np

class EulerSolver:
    def __init__(self, grid, field):
        self.grid = grid
        self.field = field
        
        # Pre-allocate flux arrays to avoid garbage collection every step
        # 4 equations: rho, rhou, rhov, rhoE
        self.flux_rho  = np.zeros(grid.num_faces)
        self.flux_rhou = np.zeros(grid.num_faces)
        self.flux_rhov = np.zeros(grid.num_faces)
        self.flux_rhoE = np.zeros(grid.num_faces)

    def compute_time_step(self, cfl=0.5):
        """
        Calculates the maximum stable dt based on the CFL condition.
        dt = CFL * min( Volume / ( (|u|+c) * Area ) )
        """
        g = self.grid
        f = self.field
        
        # Sound speed c = sqrt(gamma * p / rho)
        c = np.sqrt(f.gamma * f.p / f.rho)
        
        # Velocity magnitude
        vel_mag = np.sqrt(f.u**2 + f.v**2)
        
        # Max wave speed (approximate)
        lambda_max = vel_mag + c
        
        # We need a characteristic length scale for each cell.
        # Approximation: Length ~ sqrt(Volume)
        length_scale = np.sqrt(g.cell_volumes)
        
        # Local dt allowed
        local_dt = length_scale / (lambda_max + 1e-12)
        
        # Global dt is the minimum of all local limits
        dt = cfl * np.min(local_dt)
        return dt

    def compute_fluxes(self):
        """
        Computes fluxes across all faces using the Rusanov (Lax-Friedrichs) method.
        """
        g = self.grid
        f = self.field
        
        # 1. Identify Left (Owner) and Right (Neighbor) Indices
        idx_L = g.face_owner
        idx_R = g.face_neighbor
        
        # ---------------------------------------------------------
        # A. BOUNDARY CONDITIONS (Ghost State Approximation)
        # ---------------------------------------------------------
        # For faces where Neighbor is -1, we need to determine the "Right" state.
        # For now, we use a simple "Copy" boundary (Neumann / Extrapolation).
        # This acts like a supersonic outlet (lets flow leave).
        # Ideally, we handle Walls/Inlets specifically here.
        
        # Create temporary arrays for State_Right that default to State_Left
        # Then we only overwrite the valid neighbors.
        rho_R  = f.rho[idx_L].copy()
        u_R    = f.u[idx_L].copy()
        v_R    = f.v[idx_L].copy()
        p_R    = f.p[idx_L].copy()
        rhoE_R = f.rhoE[idx_L].copy()
        
        # Overwrite internal faces (where neighbor != -1)
        internal_mask = (idx_R != -1)
        valid_neighbors = idx_R[internal_mask]
        
        rho_R[internal_mask]  = f.rho[valid_neighbors]
        u_R[internal_mask]    = f.u[valid_neighbors]
        v_R[internal_mask]    = f.v[valid_neighbors]
        p_R[internal_mask]    = f.p[valid_neighbors]
        rhoE_R[internal_mask] = f.rhoE[valid_neighbors]
        
        # Left State is always the Owner
        rho_L  = f.rho[idx_L]
        u_L    = f.u[idx_L]
        v_L    = f.v[idx_L]
        p_L    = f.p[idx_L]
        rhoE_L = f.rhoE[idx_L]
        
        # ---------------------------------------------------------
        # B. FLUX CALCULATION (Rusanov)
        # ---------------------------------------------------------
        # Normal vectors (nx, ny) and Face Areas
        nx = g.face_normals_x
        ny = g.face_normals_y
        area = g.face_areas
        
        # 1. Compute Enthalpy: H = (rhoE + p) / rho
        H_L = (rhoE_L + p_L) / rho_L
        H_R = (rhoE_R + p_R) / rho_R
        
        # 2. Compute Normal Velocity: un = u*nx + v*ny
        un_L = u_L * nx + v_L * ny
        un_R = u_R * nx + v_R * ny
        
        # 3. Compute Physical Flux Vectors F(U)
        # Mass: rho * un
        flux_rho_L = rho_L * un_L
        flux_rho_R = rho_R * un_R
        
        # Momentum X: rho * u * un + p * nx
        flux_rhou_L = rho_L * u_L * un_L + p_L * nx
        flux_rhou_R = rho_R * u_R * un_R + p_R * nx
        
        # Momentum Y: rho * v * un + p * ny
        flux_rhov_L = rho_L * v_L * un_L + p_L * ny
        flux_rhov_R = rho_R * v_R * un_R + p_R * ny
        
        # Energy: rhoH * un
        flux_rhoE_L = rho_L * H_L * un_L
        flux_rhoE_R = rho_R * H_R * un_R
        
        # 4. Dissipation Term (The "Stabilizer")
        # Max wave speed at the interface: lambda = |un| + c
        c_L = np.sqrt(f.gamma * p_L / rho_L)
        c_R = np.sqrt(f.gamma * p_R / rho_R)
        lambda_L = np.abs(un_L) + c_L
        lambda_R = np.abs(un_R) + c_R
        alpha = np.maximum(lambda_L, lambda_R)
        
        # Rusanov Flux Formula:
        # F_face = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)
        
        self.flux_rho  = 0.5 * (flux_rho_L  + flux_rho_R)  - 0.5 * alpha * (rho_R  - rho_L)
        self.flux_rhou = 0.5 * (flux_rhou_L + flux_rhou_R) - 0.5 * alpha * (rho_R * u_R - rho_L * u_L)
        self.flux_rhov = 0.5 * (flux_rhov_L + flux_rhov_R) - 0.5 * alpha * (rho_R * v_R - rho_L * v_L)
        self.flux_rhoE = 0.5 * (flux_rhoE_L + flux_rhoE_R) - 0.5 * alpha * (rhoE_R - rhoE_L)
        
        # Multiply by Face Area (needed for the update step)
        self.flux_rho  *= area
        self.flux_rhou *= area
        self.flux_rhov *= area
        self.flux_rhoE *= area

    def update_field(self, dt):
        """
        Applies the fluxes to update the conservative variables.
        U_new = U_old - (dt / Volume) * sum(Fluxes)
        """
        g = self.grid
        f = self.field
        
        # We need to accumulate fluxes for every cell.
        # Since face_owner and face_neighbor are irregular, we use np.add.at
        
        # Initialize net flux for each cell to 0
        d_rho  = np.zeros(g.num_cells)
        d_rhou = np.zeros(g.num_cells)
        d_rhov = np.zeros(g.num_cells)
        d_rhoE = np.zeros(g.num_cells)
        
        # 1. Subtract Flux from Owners (Normal points OUT)
        np.add.at(d_rho,  g.face_owner, -self.flux_rho)
        np.add.at(d_rhou, g.face_owner, -self.flux_rhou)
        np.add.at(d_rhov, g.face_owner, -self.flux_rhov)
        np.add.at(d_rhoE, g.face_owner, -self.flux_rhoE)
        
        # 2. Add Flux to Neighbors (Normal points IN to neighbor)
        # Only for internal faces (neighbor != -1)
        mask = (g.face_neighbor != -1)
        valid_neigh = g.face_neighbor[mask]
        
        np.add.at(d_rho,  valid_neigh, self.flux_rho[mask])
        np.add.at(d_rhou, valid_neigh, self.flux_rhou[mask])
        np.add.at(d_rhov, valid_neigh, self.flux_rhov[mask])
        np.add.at(d_rhoE, valid_neigh, self.flux_rhoE[mask])
        
        # 3. Apply Update
        # U += dt/V * NetFlux
        factor = dt / g.cell_volumes
        
        f.rho  += factor * d_rho
        f.rhou += factor * d_rhou
        f.rhov += factor * d_rhov
        f.rhoE += factor * d_rhoE
        
        # 4. Update Primitives for next step
        f.primitives_from_conservatives()