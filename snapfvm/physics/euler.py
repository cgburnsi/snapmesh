"""
snapfvm/physics/euler.py
------------------------
Euler Equations with Face-Flux BCs.
"""
import numpy as np
from .base import PhysicsModel

class Euler2D(PhysicsModel):
    def __init__(self, gamma=1.4):
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
        # ... (Rusanov logic - omitted for brevity, keep previous implementation) ...
        # I will re-paste the condensed version to ensure file completeness
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
        max_speed = max(abs(vn_L/area)+c_L, abs(vn_R/area)+c_R) * area # Scale by area
        
        return 0.5*(FL+FR) - 0.5*max_speed*(q_R - q_L)

    def compute_boundary_flux(self, q_L, normal, boundary_name):
        """
        Calculates flux directly at the boundary face.
        """
        # 1. Decode Primitive Variables
        rho = q_L[0]
        u = q_L[1] / rho
        v = q_L[2] / rho
        p = (self.gamma - 1.0) * (q_L[3] - 0.5 * rho * (u**2 + v**2))
        
        nx, ny = normal # Area weighted normal
        
        # 2. Logic Switch
        if "wall" in boundary_name.lower():
            # SOLID WALL (Slip Condition for Euler)
            # Mass Flux = 0
            # Momentum Flux = Pressure * Normal
            # Energy Flux = 0 (Adiabatic/Slip work is zero since Vn=0)
            
            f_mass = 0.0
            f_mom_x = p * nx
            f_mom_y = p * ny
            f_energy = 0.0
            
            return np.array([f_mass, f_mom_x, f_mom_y, f_energy])
            
        elif "inlet" in boundary_name.lower() or "outlet" in boundary_name.lower():
            # Open Boundary (Simple Supersonic Outflow approximation)
            # F = F(Q_internal) -> "Zero Gradient"
            vn = u * nx + v * ny
            return np.array([
                rho * vn,
                q_L[1] * vn + p * nx,
                q_L[2] * vn + p * ny,
                (q_L[3] + p) * vn
            ])
            
        else:
            # Fallback (same as open)
            vn = u * nx + v * ny
            return np.array([
                rho * vn,
                q_L[1] * vn + p * nx,
                q_L[2] * vn + p * ny,
                (q_L[3] + p) * vn
            ])