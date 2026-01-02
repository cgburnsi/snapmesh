"""
snapfvm/physics/euler.py
------------------------
Euler Equations.
UPDATED: Implements Dirichlet (Inlet) and Neumann (Outlet).
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
        inv_rho = 1.0 / np.maximum(rho, 1e-12)
        u = q[..., 1] * inv_rho
        v = q[..., 2] * inv_rho
        E = q[..., 3]
        p = (self.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
        
        if q.ndim == 1:
            return np.array([rho, u, v, p])
        return np.column_stack([rho, u, v, p])

    def compute_flux(self, q_L, q_R, normal):
        # --- Standard Rusanov Flux (Same as before) ---
        rho_L, rho_R = q_L[0], q_R[0]
        u_L, v_L = q_L[1]/rho_L, q_L[2]/rho_L
        u_R, v_R = q_R[1]/rho_R, q_R[2]/rho_R
        p_L = (self.gamma-1)*(q_L[3] - 0.5*rho_L*(u_L**2+v_L**2))
        p_R = (self.gamma-1)*(q_R[3] - 0.5*rho_R*(u_R**2+v_R**2))
        
        nx, ny = normal
        area = np.sqrt(nx**2+ny**2) + 1e-16
        vn_L = u_L*nx + v_L*ny
        vn_R = u_R*nx + v_R*ny
        
        FL = np.array([rho_L*vn_L, q_L[1]*vn_L+p_L*nx, q_L[2]*vn_L+p_L*ny, (q_L[3]+p_L)*vn_L])
        FR = np.array([rho_R*vn_R, q_R[1]*vn_R+p_R*nx, q_R[2]*vn_R+p_R*ny, (q_R[3]+p_R)*vn_R])
        
        c_L = np.sqrt(self.gamma*p_L/rho_L)
        c_R = np.sqrt(self.gamma*p_R/rho_R)
        max_speed = max(abs(vn_L/area)+c_L, abs(vn_R/area)+c_R) * area 
        
        return 0.5*(FL+FR) - 0.5*max_speed*(q_R - q_L)

    def compute_boundary_flux(self, q_L, normal, boundary_name):
        """
        Handles Wall (Reflective), Inlet (Dirichlet), Outlet (Neumann).
        """
        # Decode geometry
        nx, ny = normal
        
        # 1. WALL (Slip / Reflective)
        if "wall" in boundary_name.lower():
            # Get Pressure from internal cell
            rho = q_L[0]
            u, v = q_L[1]/rho, q_L[2]/rho
            p = (self.gamma - 1.0) * (q_L[3] - 0.5 * rho * (u**2 + v**2))
            
            # Momentum Flux = Pressure * Normal
            return np.array([0.0, p*nx, p*ny, 0.0])
            
        # 2. INLET (Dirichlet / Fixed Value)
        elif "inlet" in boundary_name.lower():
            # Retrieve the fixed state we set in the script
            # If not set, default to q_L (which is wrong, but safe)
            q_fixed = self.boundary_values.get(boundary_name, q_L)
            
            # We treat the boundary as an internal face between q_L and q_fixed
            return self.compute_flux(q_L, q_fixed, normal)
            
        # 3. OUTLET (Neumann / Zero Gradient)
        elif "outlet" in boundary_name.lower():
            # Zero Gradient means q_ghost = q_internal
            # So we flux q_L against q_L
            return self.compute_flux(q_L, q_L, normal)
            
        else:
            # Fallback (Treat as outlet)
            return self.compute_flux(q_L, q_L, normal)