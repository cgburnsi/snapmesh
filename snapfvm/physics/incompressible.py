"""
snapfvm/physics/incompressible.py
---------------------------------
Incompressible Navier-Stokes using Artificial Compressibility (AC).
Variables: [p, u, v]
"""
import numpy as np
from .base import PhysicsModel

class IncompressibleAC(PhysicsModel):
    def __init__(self, rho=1.0, mu=0.01, beta=1.0):
        super().__init__()
        self.rho = rho      # Density (Constant)
        self.mu = mu        # Dynamic Viscosity
        self.beta = beta    # Artificial Sound Speed (Controls convergence)

    @property
    def num_variables(self):
        return 3 # p, u, v

    @property
    def has_viscous_terms(self):
        return True # Enables gradient calculation in Solver

    def q_to_primitives(self, q):
        # State is already [p, u, v]
        # We just return it for visualization consistency
        return q

    def compute_flux(self, q_L, q_R, normal):
        """
        Inviscid (Convective) Flux using Rusanov.
        F(Q) = [ beta^2 * vn, 
                 u*vn + p*nx/rho, 
                 v*vn + p*ny/rho ]
        """
        # 1. Unpack
        p_L, u_L, v_L = q_L[0], q_L[1], q_L[2]
        p_R, u_R, v_R = q_R[0], q_R[1], q_R[2]
        
        nx, ny = normal
        area = np.sqrt(nx**2 + ny**2) + 1e-16
        
        # 2. Normal Velocities (scaled by area)
        vn_L = u_L * nx + v_L * ny
        vn_R = u_R * nx + v_R * ny
        
        # 3. Flux Vectors F(Q)
        # Mass: beta^2 * div(u) -> Flux is beta^2 * (u.n)
        # Mom:  div(u u) + grad(p)
        
        FL = np.array([
            (self.beta**2) * vn_L,
            u_L * vn_L + p_L * nx / self.rho,
            v_L * vn_L + p_L * ny / self.rho
        ])
        
        FR = np.array([
            (self.beta**2) * vn_R,
            u_R * vn_R + p_R * nx / self.rho,
            v_R * vn_R + p_R * ny / self.rho
        ])
        
        # 4. Rusanov Wave Speed
        # Max eigenvalue = |u.n| + c, where c = beta
        # Note: vn is scaled by area, so actual velocity is vn/area
        lambda_L = abs(vn_L/area) + self.beta
        lambda_R = abs(vn_R/area) + self.beta
        max_speed = max(lambda_L, lambda_R) * area
        
        return 0.5 * (FL + FR) - 0.5 * max_speed * (q_R - q_L)

    def compute_viscous_flux(self, q_L, q_R, g_L, g_R, normal):
        """
        Diffusive Flux (Laplacian of Velocity).
        F_visc = [ 0, 
                   mu * du/dn, 
                   mu * dv/dn ]
        """
        # Simple "Two-Point" approximation for robust diffusion on unstructured grids
        # grad(phi) . n  ~=  (phi_R - phi_L) / distance
        
        # We need the distance between cell centers?
        # The 'normal' vector magnitude is the Area.
        # We don't strictly have the geometric distance d_LR passed in here.
        # APPROXIMATION: Assume standard scaling or use gradients.
        
        # Better method using passed Gradients (Green-Gauss):
        # Average Gradient at face
        grad_face = 0.5 * (g_L + g_R) # [3, 2]
        
        nx, ny = normal # Area vector
        
        # Stress Tensor (Simplified for Incompressible: tau = mu * grad(u))
        # We need tau . n
        # x-momentum flux: mu * (du/dx * nx + du/dy * ny)
        # y-momentum flux: mu * (dv/dx * nx + dv/dy * ny)
        
        # grad_face[1] is grad(u), grad_face[2] is grad(v)
        du_dn = grad_face[1, 0] * nx + grad_face[1, 1] * ny
        dv_dn = grad_face[2, 0] * nx + grad_face[2, 1] * ny
        
        # Note: Viscous flux is SUBTRACTED in the solver.
        return np.array([
            0.0,
            self.mu * du_dn,
            self.mu * dv_dn
        ])

    def compute_boundary_flux(self, q_L, normal, boundary_name):
        """
        Walls: No-Slip (u=0 or u=wall)
        """
        # 1. Determine Wall Velocity
        u_wall, v_wall = 0.0, 0.0
        
        # Check for Moving Wall (Lid)
        if "moving" in boundary_name.lower():
            # Check preset values, or default to u=1.0 (Lid)
            vals = self.boundary_values.get(boundary_name, [0, 1.0, 0])
            u_wall, v_wall = vals[1], vals[2]
            
        # 2. Ghost State Construction
        # We want u_face = u_wall.
        # q_face = 0.5*(q_L + q_ghost) => q_ghost = 2*q_face - q_L
        
        # For Rusanov flux, we feed it a Ghost State q_R
        q_R = q_L.copy()
        q_R[1] = 2*u_wall - q_L[1]
        q_R[2] = 2*v_wall - q_L[2]
        # Pressure: Zero Gradient (Neumann) -> p_R = p_L
        q_R[0] = q_L[0] 
        
        # 3. Compute Fluxes
        # Convective
        flux_conv = self.compute_flux(q_L, q_R, normal)
        
        # Viscous (One-sided gradient)
        # du/dn approx (u_wall - u_L) / (distance_to_face)
        # We don't have distance easily available here without geometry.
        # Fallback: Use the internal gradient projected.
        # (This is less accurate at walls but stable).
        # Better: Since we know u_wall, we can form a strong gradient.
        # Let's assume dist ~ sqrt(Area) / 2 roughly? 
        # No, let's trust the Green-Gauss gradient from q_L for now.
        
        # Actually, for no-slip, the convective flux of momentum is just Pressure forces.
        # (Since u.n = 0 at wall).
        # But for Moving Lid, u.n might not be zero if normal is weird? 
        # No, Lid is usually tangent. So u.n is 0.
        
        # Let's compute viscous term using Green-Gauss q_L gradient
        # (Pass g_L=0, g_R=0 to skip or handle manually?)
        # We can't easily compute viscous flux inside this function without gradients.
        
        # SIMPLIFICATION:
        # The Solver calls compute_boundary_flux.
        # We will return the TOTAL flux (Convective + Viscous)
        # But wait, solver adds viscous separately? 
        # Solver logic:
        #   flux = model.compute_boundary_flux(...)
        #   net_flux += flux
        # The solver does NOT add viscous flux automatically for boundaries in current code.
        
        # So we must include viscous drag here.
        # F_drag = mu * du/dn * Area
        # Approx du/dn ~ (u_wall - u_internal) / (h/2)
        # h ~ sqrt(nx^2+ny^2) for a square cell? Let's use 1.0/sqrt(Area_normal) approx.
        area = np.sqrt(normal[0]**2 + normal[1]**2)
        dist = np.sqrt(area) * 0.5 # Rough approx to face center
        
        tau_x = self.mu * (u_wall - q_L[1]) / dist
        tau_y = self.mu * (v_wall - q_L[2]) / dist
        
        # Add viscous drag to the momentum flux
        # Note: Viscous term is usually subtracted (Diffusion). 
        # But here tau is a force on the fluid?
        # Standard: d/dt = -Div(F_conv) + Div(F_visc)
        # Flux vector pointing OUT.
        # If u_wall > u_L, fluid is dragged +x. Flux of momentum entering should be positive.
        # Vector pointing out: Flux leaving is negative.
        
        flux_conv[1] -= tau_x * area
        flux_conv[2] -= tau_y * area
        
        return flux_conv