"""
snapfvm/physics/euler.py
------------------------
Compressible Euler Equations.
UPDATED: 
1. Robustness Clamping (Prevents Negative Pressure/Density).
2. API Compatibility (Accepts 'distance' arg from Solver).
"""
import numpy as np
from .base import PhysicsModel

class Euler2D(PhysicsModel):
    def __init__(self, gamma=1.4):
        super().__init__()
        self.gamma = gamma

    @property
    def num_variables(self):
        return 4 

    def q_to_primitives(self, q):
        rho = q[..., 0]
        # Safety Clamp for Density
        rho = np.maximum(rho, 1e-12)
        inv_rho = 1.0 / rho
        
        u = q[..., 1] * inv_rho
        v = q[..., 2] * inv_rho
        E = q[..., 3]
        p = (self.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
        
        # Safety Clamp for Pressure (Visualization only)
        p = np.maximum(p, 1e-12)
        
        if q.ndim == 1:
            return np.array([rho, u, v, p])
        return np.column_stack([rho, u, v, p])

    def compute_flux(self, q_L, q_R, normal):
        # --- 1. DECODE LEFT STATE ---
        rho_L = max(q_L[0], 1e-12) # Clamp Density
        u_L, v_L = q_L[1]/rho_L, q_L[2]/rho_L
        p_L = (self.gamma-1)*(q_L[3] - 0.5*rho_L*(u_L**2+v_L**2))
        p_L = max(p_L, 1e-12)      # Clamp Pressure
        
        # --- 2. DECODE RIGHT STATE ---
        rho_R = max(q_R[0], 1e-12)
        u_R, v_R = q_R[1]/rho_R, q_R[2]/rho_R
        p_R = (self.gamma-1)*(q_R[3] - 0.5*rho_R*(u_R**2+v_R**2))
        p_R = max(p_R, 1e-12)

        # --- 3. GEOMETRY ---
        nx, ny = normal
        area = np.sqrt(nx**2+ny**2) + 1e-16
        vn_L = u_L*nx + v_L*ny
        vn_R = u_R*nx + v_R*ny
        
        # --- 4. FLUXES ---
        FL = np.array([rho_L*vn_L, q_L[1]*vn_L+p_L*nx, q_L[2]*vn_L+p_L*ny, (q_L[3]+p_L)*vn_L])
        FR = np.array([rho_R*vn_R, q_R[1]*vn_R+p_R*nx, q_R[2]*vn_R+p_R*ny, (q_R[3]+p_R)*vn_R])
        
        # --- 5. DISSIPATION (Rusanov) ---
        c_L = np.sqrt(self.gamma*p_L/rho_L)
        c_R = np.sqrt(self.gamma*p_R/rho_R)
        max_speed = max(abs(vn_L/area)+c_L, abs(vn_R/area)+c_R) * area 
        
        return 0.5*(FL+FR) - 0.5*max_speed*(q_R - q_L)

    def compute_boundary_flux(self, q_L, normal, boundary_name, distance=1.0):
        """
        Handles Wall (Reflective), Inlet (Dirichlet), Outlet (Neumann).
        'distance' is accepted for API compatibility but ignored (Inviscid).
        """
        # Decode and Clamp
        nx, ny = normal
        rho = max(q_L[0], 1e-12)
        u, v = q_L[1]/rho, q_L[2]/rho
        p = (self.gamma - 1.0) * (q_L[3] - 0.5 * rho * (u**2 + v**2))
        p = max(p, 1e-12)
        
        if "wall" in boundary_name.lower():
            # Slip Wall: Momentum Flux = p * n
            return np.array([0.0, p*nx, p*ny, 0.0])
            
        elif "inlet" in boundary_name.lower():
            q_fixed = self.boundary_values.get(boundary_name, q_L)
            return self.compute_flux(q_L, q_fixed, normal)
            
        elif "outlet" in boundary_name.lower():
            # Zero Gradient
            return self.compute_flux(q_L, q_L, normal)
            
        else:
            return self.compute_flux(q_L, q_L, normal)