"""
snapfvm/numerics.py
-------------------
High-Performance Kernels using Numba JIT compilation.
This replaces the slow Python loops in solver.py.
"""
import numpy as np
from numba import njit

# fastmath=True allows the compiler to rearrange floating point ops for speed
# cache=True saves the compiled binary to disk so subsequent runs are instant
@njit(fastmath=True, cache=True)
def incompressible_step_kernel(
    n_cells, n_faces,
    face_cells, face_normals, face_midpoints, 
    cell_centers, cell_volumes,
    q, dt, 
    rho, mu, beta,
    order
):
    """
    The Compute Kernel. 
    Accepts raw NumPy arrays only. No objects.
    """
    residuals = np.zeros((n_cells, 3), dtype=np.float64)
    
    # --- 1. PRE-CALCULATE FACE AREAS ---
    # We do this inside the kernel to avoid storing another array in Python
    # face_normals is [N, 2] (scaled by area)
    
    # --- 2. GRADIENTS (Green-Gauss) ---
    grad_q = np.zeros((n_cells, 3, 2), dtype=np.float64)
    
    for i in range(n_faces):
        c_L = face_cells[i, 0]
        c_R = face_cells[i, 1]
        nx = face_normals[i, 0]
        ny = face_normals[i, 1]
        
        # Internal Face
        if c_R != -1:
            # Simple average for gradient reconstruction
            # q is [N_cells, 3]
            p_face = 0.5 * (q[c_L, 0] + q[c_R, 0])
            u_face = 0.5 * (q[c_L, 1] + q[c_R, 1])
            v_face = 0.5 * (q[c_L, 2] + q[c_R, 2])
            
            # Add to Left (Green-Gauss: sum(val * normal))
            grad_q[c_L, 0, 0] += p_face * nx
            grad_q[c_L, 0, 1] += p_face * ny
            grad_q[c_L, 1, 0] += u_face * nx
            grad_q[c_L, 1, 1] += u_face * ny
            grad_q[c_L, 2, 0] += v_face * nx
            grad_q[c_L, 2, 1] += v_face * ny
            
            # Subtract from Right (Normal points L->R, so it is negative for R)
            grad_q[c_R, 0, 0] -= p_face * nx
            grad_q[c_R, 0, 1] -= p_face * ny
            grad_q[c_R, 1, 0] -= u_face * nx
            grad_q[c_R, 1, 1] -= u_face * ny
            grad_q[c_R, 2, 0] -= v_face * nx
            grad_q[c_R, 2, 1] -= v_face * ny
            
        # Boundary Face (Use Cell Value)
        else:
            grad_q[c_L, 0, 0] += q[c_L, 0] * nx
            grad_q[c_L, 0, 1] += q[c_L, 0] * ny
            grad_q[c_L, 1, 0] += q[c_L, 1] * nx
            grad_q[c_L, 1, 1] += q[c_L, 1] * ny
            grad_q[c_L, 2, 0] += q[c_L, 2] * nx
            grad_q[c_L, 2, 1] += q[c_L, 2] * ny

    # Normalize Gradients by Cell Volume
    for c in range(n_cells):
        inv_vol = 1.0 / cell_volumes[c]
        for var in range(3):
            grad_q[c, var, 0] *= inv_vol
            grad_q[c, var, 1] *= inv_vol

    # --- 3. FLUX LOOP ---
    for i in range(n_faces):
        c_L = face_cells[i, 0]
        c_R = face_cells[i, 1]
        
        # If Boundary, skip (handled in Python for simplicity/flexibility)
        if c_R == -1:
            continue

        nx_scaled = face_normals[i, 0]
        ny_scaled = face_normals[i, 1]
        area = np.sqrt(nx_scaled**2 + ny_scaled**2) + 1e-16
        
        # Unit Normal
        nx = nx_scaled / area
        ny = ny_scaled / area
        
        # --- RECONSTRUCTION ---
        if order == 1:
            # First Order: Constant
            p_L, u_L, v_L = q[c_L, 0], q[c_L, 1], q[c_L, 2]
            p_R, u_R, v_R = q[c_R, 0], q[c_R, 1], q[c_R, 2]
        else:
            # Second Order: Linear Extrapolation
            # Left
            dx_L = face_midpoints[i, 0] - cell_centers[c_L, 0]
            dy_L = face_midpoints[i, 1] - cell_centers[c_L, 1]
            
            p_L = q[c_L, 0] + (grad_q[c_L, 0, 0]*dx_L + grad_q[c_L, 0, 1]*dy_L)
            u_L = q[c_L, 1] + (grad_q[c_L, 1, 0]*dx_L + grad_q[c_L, 1, 1]*dy_L)
            v_L = q[c_L, 2] + (grad_q[c_L, 2, 0]*dx_L + grad_q[c_L, 2, 1]*dy_L)
            
            # Right
            dx_R = face_midpoints[i, 0] - cell_centers[c_R, 0]
            dy_R = face_midpoints[i, 1] - cell_centers[c_R, 1]
            
            p_R = q[c_R, 0] + (grad_q[c_R, 0, 0]*dx_R + grad_q[c_R, 0, 1]*dy_R)
            u_R = q[c_R, 1] + (grad_q[c_R, 1, 0]*dx_R + grad_q[c_R, 1, 1]*dy_R)
            v_R = q[c_R, 2] + (grad_q[c_R, 2, 0]*dx_R + grad_q[c_R, 2, 1]*dy_R)

        # --- FLUX CALCULATION (Incompressible AC) ---
        vn_L = u_L * nx + v_L * ny
        vn_R = u_R * nx + v_R * ny

        # Flux L
        # Mass: beta^2 * vn
        # Mom X: u*vn + p*nx/rho
        # Mom Y: v*vn + p*ny/rho
        FL_0 = (beta**2) * vn_L
        FL_1 = u_L * vn_L + p_L * nx / rho
        FL_2 = v_L * vn_L + p_L * ny / rho

        # Flux R
        FR_0 = (beta**2) * vn_R
        FR_1 = u_R * vn_R + p_R * nx / rho
        FR_2 = v_R * vn_R + p_R * ny / rho

        # Rusanov Wave Speed
        # lambda = |vn| + beta
        lambda_L = np.abs(vn_L) + beta
        lambda_R = np.abs(vn_R) + beta
        max_speed = max(lambda_L, lambda_R)

        # Rusanov Flux: 0.5(FL + FR) - 0.5*c*(QR - QL)
        flux_0 = 0.5 * (FL_0 + FR_0) - 0.5 * max_speed * (p_R - p_L)
        flux_1 = 0.5 * (FL_1 + FR_1) - 0.5 * max_speed * (u_R - u_L)
        flux_2 = 0.5 * (FL_2 + FR_2) - 0.5 * max_speed * (v_R - v_L)
        
        # --- VISCOUS FLUX (Simplified) ---
        # grad u . n  (Average gradient approximation)
        du_dx = 0.5 * (grad_q[c_L, 1, 0] + grad_q[c_R, 1, 0])
        du_dy = 0.5 * (grad_q[c_L, 1, 1] + grad_q[c_R, 1, 1])
        dv_dx = 0.5 * (grad_q[c_L, 2, 0] + grad_q[c_R, 2, 0])
        dv_dy = 0.5 * (grad_q[c_L, 2, 1] + grad_q[c_R, 2, 1])
        
        du_dn = du_dx * nx + du_dy * ny
        dv_dn = dv_dx * nx + dv_dy * ny
        
        visc_1 = mu * du_dn
        visc_2 = mu * dv_dn
        
        # --- TOTAL NET FLUX ---
        # Multiply by Area (since flux is per unit area)
        # flux_0 is mass, flux_1 is x-mom, flux_2 is y-mom
        
        net_0 = flux_0 * area
        net_1 = (flux_1 - visc_1) * area
        net_2 = (flux_2 - visc_2) * area
        
        # Accumulate Residuals
        # Left gets -Flux (Leaving)
        residuals[c_L, 0] -= net_0
        residuals[c_L, 1] -= net_1
        residuals[c_L, 2] -= net_2
        
        # Right gets +Flux (Entering)
        residuals[c_R, 0] += net_0
        residuals[c_R, 1] += net_1
        residuals[c_R, 2] += net_2
        
    return residuals