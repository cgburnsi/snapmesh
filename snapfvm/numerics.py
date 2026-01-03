"""
snapfvm/numerics.py
-------------------
High-Performance Kernels using Numba JIT compilation.
FORCE UPDATE: cache=False to clear broken signatures.
"""
import numpy as np
from numba import njit

# --- HELPER FUNCTIONS ---
@njit(fastmath=True, cache=False)
def minmod(a, b):
    if a * b <= 0:
        return 0.0
    if np.abs(a) < np.abs(b):
        return a
    return b

# --- INCOMPRESSIBLE KERNEL ---
@njit(fastmath=True, cache=False)
def incompressible_step_kernel(
    n_cells, n_faces,
    face_cells, face_normals, face_midpoints, 
    cell_centers, cell_volumes,
    q, dt, 
    rho, mu, beta,
    order
):
    residuals = np.zeros((n_cells, 3), dtype=np.float64)
    
    # 1. Gradients
    grad_q = np.zeros((n_cells, 3, 2), dtype=np.float64)
    for i in range(n_faces):
        # Explicit Casts
        idx_L = int(face_cells[i, 0])
        idx_R = int(face_cells[i, 1])
        nx = face_normals[i, 0]
        ny = face_normals[i, 1]
        
        if idx_R != -1:
            for v in range(3):
                avg = 0.5*(q[idx_L, v] + q[idx_R, v])
                term_x = avg * nx
                term_y = avg * ny
                grad_q[idx_L, v, 0] += term_x; grad_q[idx_L, v, 1] += term_y
                grad_q[idx_R, v, 0] -= term_x; grad_q[idx_R, v, 1] -= term_y
        else:
            for v in range(3):
                term_x = q[idx_L, v] * nx
                term_y = q[idx_L, v] * ny
                grad_q[idx_L, v, 0] += term_x; grad_q[idx_L, v, 1] += term_y

    for c in range(n_cells):
        inv_vol = 1.0/cell_volumes[c]
        grad_q[c] *= inv_vol

    # 2. Fluxes
    for i in range(n_faces):
        idx_L = int(face_cells[i, 0])
        idx_R = int(face_cells[i, 1])
        if idx_R == -1: continue

        nx_s = face_normals[i, 0]
        ny_s = face_normals[i, 1]
        area = np.sqrt(nx_s**2 + ny_s**2) + 1e-16
        nx, ny = nx_s/area, ny_s/area

        # Reconstruction
        if order == 1:
            q_L, q_R = q[idx_L], q[idx_R]
        else:
            dx_L = face_midpoints[i, 0] - cell_centers[idx_L, 0]
            dy_L = face_midpoints[i, 1] - cell_centers[idx_L, 1]
            q_L = np.zeros(3)
            for v in range(3):
                q_L[v] = q[idx_L, v] + (grad_q[idx_L, v, 0]*dx_L + grad_q[idx_L, v, 1]*dy_L)

            dx_R = face_midpoints[i, 0] - cell_centers[idx_R, 0]
            dy_R = face_midpoints[i, 1] - cell_centers[idx_R, 1]
            q_R = np.zeros(3)
            for v in range(3):
                q_R[v] = q[idx_R, v] + (grad_q[idx_R, v, 0]*dx_R + grad_q[idx_R, v, 1]*dy_R)

        vn_L = q_L[1]*nx + q_L[2]*ny
        vn_R = q_R[1]*nx + q_R[2]*ny
        
        FL = np.array([beta**2 * vn_L, q_L[1]*vn_L + q_L[0]*nx/rho, q_L[2]*vn_L + q_L[0]*ny/rho])
        FR = np.array([beta**2 * vn_R, q_R[1]*vn_R + q_R[0]*nx/rho, q_R[2]*vn_R + q_R[0]*ny/rho])
        
        max_s = max(abs(vn_L) + beta, abs(vn_R) + beta)
        flux = 0.5*(FL + FR) - 0.5*max_s*(q_R - q_L)
        
        # Viscous
        du_dx = 0.5*(grad_q[idx_L, 1, 0] + grad_q[idx_R, 1, 0])
        du_dy = 0.5*(grad_q[idx_L, 1, 1] + grad_q[idx_R, 1, 1])
        dv_dx = 0.5*(grad_q[idx_L, 2, 0] + grad_q[idx_R, 2, 0])
        dv_dy = 0.5*(grad_q[idx_L, 2, 1] + grad_q[idx_R, 2, 1])
        
        visc = np.array([0.0, mu*(du_dx*nx + du_dy*ny), mu*(dv_dx*nx + dv_dy*ny)])
        net = (flux - visc) * area
        
        residuals[idx_L] -= net
        residuals[idx_R] += net
        
    return residuals

# --- EULER KERNEL ---
@njit(fastmath=True, cache=False)
def euler_step_kernel(
    n_cells, n_faces,
    face_cells, face_normals, face_midpoints, 
    cell_centers, cell_volumes,
    q, dt, 
    gamma,
    order
):
    residuals = np.zeros((n_cells, 4), dtype=np.float64)
    
    # 1. Gradients
    grad_q = np.zeros((n_cells, 4, 2), dtype=np.float64)
    for i in range(n_faces):
        # Force integer cast and use new var names
        idx_L = int(face_cells[i, 0])
        idx_R = int(face_cells[i, 1])
        nx = face_normals[i, 0]
        ny = face_normals[i, 1]
        
        if idx_R != -1:
            for v in range(4):
                avg = 0.5*(q[idx_L, v] + q[idx_R, v])
                term_x = avg * nx
                term_y = avg * ny
                grad_q[idx_L, v, 0] += term_x; grad_q[idx_L, v, 1] += term_y
                grad_q[idx_R, v, 0] -= term_x; grad_q[idx_R, v, 1] -= term_y
        else:
            for v in range(4):
                term_x = q[idx_L, v] * nx
                term_y = q[idx_L, v] * ny
                grad_q[idx_L, v, 0] += term_x; grad_q[idx_L, v, 1] += term_y

    for c in range(n_cells):
        inv_vol = 1.0/cell_volumes[c]
        grad_q[c] *= inv_vol

    # 2. Fluxes
    for i in range(n_faces):
        idx_L = int(face_cells[i, 0])
        idx_R = int(face_cells[i, 1])
        if idx_R == -1: continue

        nx_s = face_normals[i, 0]
        ny_s = face_normals[i, 1]
        area = np.sqrt(nx_s**2 + ny_s**2) + 1e-16
        nx, ny = nx_s/area, ny_s/area

        if order == 1:
            q_L, q_R = q[idx_L], q[idx_R]
        else:
            dx_L = face_midpoints[i, 0] - cell_centers[idx_L, 0]
            dy_L = face_midpoints[i, 1] - cell_centers[idx_L, 1]
            dx_R = face_midpoints[i, 0] - cell_centers[idx_R, 0]
            dy_R = face_midpoints[i, 1] - cell_centers[idx_R, 1]
            
            q_L = np.zeros(4)
            q_R = np.zeros(4)
            for v in range(4):
                slope_L = grad_q[idx_L, v, 0]*dx_L + grad_q[idx_L, v, 1]*dy_L
                slope_R = grad_q[idx_R, v, 0]*dx_R + grad_q[idx_R, v, 1]*dy_R
                diff = q[idx_R, v] - q[idx_L, v]
                
                phi_L = minmod(slope_L, diff)
                phi_R = minmod(slope_R, diff)
                
                q_L[v] = q[idx_L, v] + phi_L
                q_R[v] = q[idx_R, v] + phi_R

        # Decode L
        rho_L = max(q_L[0], 1e-12)
        u_L = q_L[1]/rho_L; v_L = q_L[2]/rho_L
        p_L = max((gamma-1)*(q_L[3] - 0.5*rho_L*(u_L**2+v_L**2)), 1e-12)
        c_L = np.sqrt(gamma*p_L/rho_L)
        
        # Decode R
        rho_R = max(q_R[0], 1e-12)
        u_R = q_R[1]/rho_R; v_R = q_R[2]/rho_R
        p_R = max((gamma-1)*(q_R[3] - 0.5*rho_R*(u_R**2+v_R**2)), 1e-12)
        c_R = np.sqrt(gamma*p_R/rho_R)
        
        vn_L = u_L*nx + v_L*ny
        vn_R = u_R*nx + v_R*ny
        
        FL = np.array([rho_L*vn_L, q_L[1]*vn_L + p_L*nx, q_L[2]*vn_L + p_L*ny, (q_L[3]+p_L)*vn_L])
        FR = np.array([rho_R*vn_R, q_R[1]*vn_R + p_R*nx, q_R[2]*vn_R + p_R*ny, (q_R[3]+p_R)*vn_R])
        
        max_s = max(abs(vn_L)+c_L, abs(vn_R)+c_R)
        
        flux = 0.5*(FL + FR) - 0.5*max_s*(q_R - q_L)
        net = flux * area
        
        # USAGE SITE CASTS are implicitly handled by using idx_L (which is int)
        residuals[idx_L] -= net
        residuals[idx_R] += net
        
    return residuals