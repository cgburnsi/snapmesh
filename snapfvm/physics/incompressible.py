"""
snapfvm/physics/incompressible.py
"""
import numpy as np
from .base import PhysicsModel

class IncompressibleAC(PhysicsModel):
    def __init__(self, rho=1.0, mu=0.01, beta=1.0):
        super().__init__()
        self.rho = rho 
        self.mu = mu
        self.beta = beta

    @property
    def num_variables(self):
        return 3

    @property
    def has_viscous_terms(self):
        return True

    def q_to_primitives(self, q):
        return q

    def compute_flux(self, q_L, q_R, normal):
        # ... (Same as before) ...
        p_L, u_L, v_L = q_L[0], q_L[1], q_L[2]
        p_R, u_R, v_R = q_R[0], q_R[1], q_R[2]
        
        nx, ny = normal
        area = np.sqrt(nx**2 + ny**2) + 1e-16
        
        vn_L = u_L * nx + v_L * ny
        vn_R = u_R * nx + v_R * ny
        
        FL = np.array([(self.beta**2)*vn_L, u_L*vn_L + p_L*nx/self.rho, v_L*vn_L + p_L*ny/self.rho])
        FR = np.array([(self.beta**2)*vn_R, u_R*vn_R + p_R*nx/self.rho, v_R*vn_R + p_R*ny/self.rho])
        
        lambda_L = abs(vn_L/area) + self.beta
        lambda_R = abs(vn_R/area) + self.beta
        max_speed = max(lambda_L, lambda_R) * area
        
        return 0.5 * (FL + FR) - 0.5 * max_speed * (q_R - q_L)

    def compute_viscous_flux(self, q_L, q_R, g_L, g_R, normal):
        # ... (Same as before) ...
        grad_face = 0.5 * (g_L + g_R)
        du_dn = grad_face[1, 0] * normal[0] + grad_face[1, 1] * normal[1]
        dv_dn = grad_face[2, 0] * normal[0] + grad_face[2, 1] * normal[1]
        
        return np.array([0.0, self.mu * du_dn, self.mu * dv_dn])

    def compute_boundary_flux(self, q_L, normal, boundary_name, distance=1.0):
        """
        Calculates Wall Flux using Exact Geometric Distance for Shear Stress.
        """
        # 1. Wall State
        u_wall, v_wall = 0.0, 0.0
        if "moving" in boundary_name.lower():
            vals = self.boundary_values.get(boundary_name, [0, 1.0, 0])
            u_wall, v_wall = vals[1], vals[2]
            
        # 2. Ghost State (Convection)
        q_R = q_L.copy()
        q_R[1] = 2*u_wall - q_L[1]
        q_R[2] = 2*v_wall - q_L[2]
        
        flux_conv = self.compute_flux(q_L, q_R, normal)
        
        # 3. Viscous Drag (Shear Stress)
        # Gradient = (Wall_Val - Center_Val) / Distance_Perpendicular
        
        # 'normal' contains Area (Edge Length). 
        # Flux = Stress * Area.
        # Stress = mu * du/dn
        
        area = np.sqrt(normal[0]**2 + normal[1]**2) + 1e-16
        
        # du/dn approx
        du_dn = (u_wall - q_L[1]) / distance
        dv_dn = (v_wall - q_L[2]) / distance
        
        # Viscous Flux Vector
        # We multiply by Area because 'normal' in compute_flux scaled things by Area?
        # No, compute_flux returns Total Flux (FluxDensity * Area).
        # So we must return Total Viscous Force.
        
        flux_visc = np.array([
            0.0,
            self.mu * du_dn * area,
            self.mu * dv_dn * area
        ])
        
        return flux_conv - flux_visc