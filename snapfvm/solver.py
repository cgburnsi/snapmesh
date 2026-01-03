"""
snapfvm/solver.py
-----------------
Generic FVM Solver.
UPDATED: Adds Positivity Clamping to prevent crash on startup.
"""
import numpy as np
from .numerics import incompressible_step_kernel, euler_step_kernel
from .physics.euler import Euler2D
from .physics.incompressible import IncompressibleAC

class FiniteVolumeSolver:
    def __init__(self, grid, physics_model, order=1):
        self.grid = grid
        self.model = physics_model
        self.order = order
        self.q = np.zeros((grid.n_cells, self.model.num_variables), dtype=float)
        
    def set_initial_condition(self, q_init):
        self.q[:] = q_init

    def step(self, dt):
        # 1. Cast indices for Numba
        face_cells_int = self.grid.face_cells.astype(np.int64)

        # 2. Run Kernel (Internal Fluxes)
        if isinstance(self.model, Euler2D):
            residuals = euler_step_kernel(
                self.grid.n_cells, self.grid.n_faces,
                face_cells_int,
                self.grid.face_normals, self.grid.face_midpoints, 
                self.grid.cell_centers, self.grid.cell_volumes,
                self.q, dt,
                self.model.gamma,
                self.order
            )
        elif isinstance(self.model, IncompressibleAC):
            residuals = incompressible_step_kernel(
                self.grid.n_cells, self.grid.n_faces,
                face_cells_int,
                self.grid.face_normals, self.grid.face_midpoints, 
                self.grid.cell_centers, self.grid.cell_volumes,
                self.q, dt,
                self.model.rho, self.model.mu, self.model.beta,
                self.order
            )
        else:
            raise ValueError("Physics Model not supported.")

        # 3. Boundary Conditions (Python)
        for i in range(self.grid.n_faces):
            c_right = int(self.grid.face_cells[i, 1])
            if c_right == -1: 
                c_left = int(self.grid.face_cells[i, 0])
                normal = self.grid.face_normals[i]
                q_L = self.q[c_left]
                
                # Geometric distance (needed for viscous walls)
                f_mid = self.grid.face_midpoints[i]
                c_cent = self.grid.cell_centers[c_left]
                dist = np.sqrt(np.sum((f_mid - c_cent)**2))
                dist = max(dist, 1e-12)
                
                group_id = self.grid.face_groups[i]
                bc_name = self.grid.group_names.get(group_id, "unknown")
                
                # Compute Flux
                try:
                    flux = self.model.compute_boundary_flux(q_L, normal, bc_name, distance=dist)
                    residuals[c_left] -= flux
                except RuntimeWarning:
                    pass # Ignore warnings here, clamping will catch bad values later

        # 4. Update Solution
        self.q += (dt / self.grid.cell_volumes[:, None]) * residuals
        
        # 5. SAFETY CLAMP (Positivity Preservation)
        # This prevents NaN/Overflow by forcing rho and p to stay positive.
        if isinstance(self.model, Euler2D):
            # Clamp Density
            self.q[:, 0] = np.maximum(self.q[:, 0], 1e-4)
            
            # Reconstruct Pressure to Clamp Energy
            rho = self.q[:, 0]
            u = self.q[:, 1] / rho
            v = self.q[:, 2] / rho
            E = self.q[:, 3]
            p = (self.model.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
            
            # If Pressure is negative, add energy to fix it
            bad_p = p < 1e-4
            if np.any(bad_p):
                # Set p to min value
                target_p = 1e-4
                # E = p/(g-1) + 0.5*rho*v^2
                new_E = target_p / (self.model.gamma - 1.0) + 0.5 * rho[bad_p] * (u[bad_p]**2 + v[bad_p]**2)
                self.q[bad_p, 3] = new_E

        return np.max(np.abs(residuals))