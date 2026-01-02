"""
snapfvm/physics/euler.py
------------------------
Compressible Euler Equations.
Implements the PhysicsModel interface.
"""
import numpy as np
from .base import PhysicsModel

class Euler2D(PhysicsModel):
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    @property
    def num_variables(self):
        return 4 # rho, rho*u, rho*v, rho*E

    @property
    def has_viscous_terms(self):
        return False # Euler is inviscid

    def q_to_primitives(self, q):
        """ Decodes Q -> [rho, u, v, p] """
        # Handle both 1D (single cell) and 2D (entire grid) arrays
        rho = q[..., 0]
        inv_rho = 1.0 / np.maximum(rho, 1e-12)
        
        u = q[..., 1] * inv_rho
        v = q[..., 2] * inv_rho
        
        E = q[..., 3]
        ke = 0.5 * rho * (u**2 + v**2)
        p = (self.gamma - 1.0) * (E - ke)
        
        if q.ndim == 1:
            return np.array([rho, u, v, p])
        return np.column_stack([rho, u, v, p])

    def compute_flux(self, q_L, q_R, normal):
        """
        Rusanov (Local Lax-Friedrichs) Flux.
        F_face = 0.5*(F_L + F_R) - 0.5*max_wave_speed*(Q_R - Q_L)
        """
        # 1. Primitives L
        rho_L = q_L[0]
        inv_rho_L = 1.0 / max(rho_L, 1e-12)
        u_L, v_L = q_L[1]*inv_rho_L, q_L[2]*inv_rho_L
        p_L = (self.gamma - 1.0) * (q_L[3] - 0.5*rho_L*(u_L**2 + v_L**2))
        
        # 2. Primitives R
        rho_R = q_R[0]
        inv_rho_R = 1.0 / max(rho_R, 1e-12)
        u_R, v_R = q_R[1]*inv_rho_R, q_R[2]*inv_rho_R
        p_R = (self.gamma - 1.0) * (q_R[3] - 0.5*rho_R*(u_R**2 + v_R**2))
        
        # 3. Geometry
        # 'normal' includes Area magnitude. We need Unit Normal for projection.
        nx, ny = normal
        area = np.sqrt(nx**2 + ny**2) + 1e-16
        inx, iny = nx/area, ny/area
        
        # 4. Normal Velocities
        vn_L = u_L*nx + v_L*ny # Not unit! This is (u,v) dot AreaVec
        vn_R = u_R*nx + v_R*ny
        
        # 5. Flux Vectors F(Q)
        # F = [rho*Vn, rho*u*Vn + p*nx, rho*v*Vn + p*ny, (E+p)*Vn]
        FL = np.array([
            rho_L * vn_L,
            q_L[1] * vn_L + p_L * nx,
            q_L[2] * vn_L + p_L * ny,
            (q_L[3] + p_L) * vn_L
        ])
        
        FR = np.array([
            rho_R * vn_R,
            q_R[1] * vn_R + p_R * nx,
            q_R[2] * vn_R + p_R * ny,
            (q_R[3] + p_R) * vn_R
        ])
        
        # 6. Dissipation (Wave Speed)
        c_L = np.sqrt(self.gamma * max(p_L, 1e-6) / rho_L)
        c_R = np.sqrt(self.gamma * max(p_R, 1e-6) / rho_R)
        
        # Max wave speed (|u.n| + c) projected on normal direction
        # We need real velocity (u,v) projected onto unit normal (inx, iny)
        speed_L = abs(u_L*inx + v_L*iny) + c_L
        speed_R = abs(u_R*inx + v_R*iny) + c_R
        max_speed = max(speed_L, speed_R)
        
        dissipation = 0.5 * max_speed * area * (q_R - q_L)
        
        return 0.5 * (FL + FR) - dissipation