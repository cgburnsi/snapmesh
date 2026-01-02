"""
snapfvm/solver.py
-----------------
Generic FVM Solver.
UPDATED: Uses Numba Kernel for performance.
"""
import numpy as np
from .numerics import incompressible_step_kernel

class FiniteVolumeSolver:
    def __init__(self, grid, physics_model, order=1):
        self.grid = grid
        self.model = physics_model
        self.order = order # 1 or 2
        
        self.q = np.zeros((grid.n_cells, self.model.num_variables), dtype=float)
        
        print(f"--- FVM Solver Initialized ---")
        print(f"   -> Physics: {self.model.__class__.__name__}")
        print(f"   -> Backend: Numba Accelerated")

    def set_initial_condition(self, q_init):
        self.q[:] = q_init

    def step(self, dt):
        # 1. Run HIGH SPEED KERNEL for Internal Faces
        # This handles Gradient Reconstruction + Flux + Viscosity for all internal edges
        residuals = incompressible_step_kernel(
            self.grid.n_cells, self.grid.n_faces,
            self.grid.face_cells, self.grid.face_normals, 
            self.grid.face_midpoints, self.grid.cell_centers, self.grid.cell_volumes,
            self.q, dt,
            self.model.rho, self.model.mu, self.model.beta,
            self.order
        )
        
        # 2. Handle Boundaries in Python
        # Boundaries are complex and few, so we keep them in flexible Python.
        # This is efficient enough (O(sqrt(N))).
        for i in range(self.grid.n_faces):
            c_right = self.grid.face_cells[i, 1]
            if c_right == -1: # Boundary
                c_left = self.grid.face_cells[i, 0]
                normal = self.grid.face_normals[i]
                
                # Use Cell Value (First Order at Wall for stability)
                q_L = self.q[c_left]
                
                group_id = self.grid.face_groups[i]
                bc_name = self.grid.group_names.get(group_id, "unknown")
                
                flux = self.model.compute_boundary_flux(q_L, normal, bc_name)
                
                residuals[c_left] -= flux

        # 3. Update State
        # q_new = q_old + (dt/vol) * Residuals
        self.q += (dt / self.grid.cell_volumes[:, None]) * residuals
        
        return np.max(np.abs(residuals))