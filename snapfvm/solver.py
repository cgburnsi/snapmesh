"""
snapfvm/solver.py
"""
import numpy as np
from .numerics import incompressible_step_kernel

class FiniteVolumeSolver:
    def __init__(self, grid, physics_model, order=1):
        self.grid = grid
        self.model = physics_model
        self.order = order
        self.q = np.zeros((grid.n_cells, self.model.num_variables), dtype=float)
        
        # Cache for Numba
        self.face_areas = np.sqrt(grid.face_normals[:,0]**2 + grid.face_normals[:,1]**2)

    def set_initial_condition(self, q_init):
        self.q[:] = q_init

    def step(self, dt):
        # 1. Numba Kernel (Internal)
        residuals = incompressible_step_kernel(
            self.grid.n_cells, self.grid.n_faces,
            self.grid.face_cells, self.grid.face_normals, self.grid.face_midpoints, 
            self.grid.cell_centers, self.grid.cell_volumes,
            self.q, dt,
            self.model.rho, self.model.mu, self.model.beta,
            self.order
        )
        
        # 2. Python Boundary Loop
        for i in range(self.grid.n_faces):
            c_right = self.grid.face_cells[i, 1]
            if c_right == -1: # Boundary
                c_left = self.grid.face_cells[i, 0]
                normal = self.grid.face_normals[i]
                q_L = self.q[c_left]
                
                # --- GEOMETRIC CORRECTION ---
                # Calculate exact distance from Cell Center to Face Center
                f_mid = self.grid.face_midpoints[i]
                c_cent = self.grid.cell_centers[c_left]
                
                # Vector d
                d_vec = f_mid - c_cent
                
                # Project onto normal? 
                # Strict Finite Volume definition uses orthogonal distance (d . n)
                # But for general unstructured, simple Euclidean dist is often robust enough 
                # and handles skewed cells better in simplified flux schemes.
                # Let's use Euclidean distance.
                dist = np.sqrt(np.dot(d_vec, d_vec))
                
                # Safety
                dist = max(dist, 1e-12)
                
                group_id = self.grid.face_groups[i]
                bc_name = self.grid.group_names.get(group_id, "unknown")
                
                # Pass dist to model
                flux = self.model.compute_boundary_flux(q_L, normal, bc_name, distance=dist)
                
                residuals[c_left] -= flux

        # 3. Update
        self.q += (dt / self.grid.cell_volumes[:, None]) * residuals
        return np.max(np.abs(residuals))