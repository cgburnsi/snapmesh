"""
snapfvm/limiters.py
-------------------
Slope Limiters for Second-Order Reconstruction on Unstructured Grids.
Implements Barth-Jespersen limiting to prevent oscillations (Gibbs phenomenon).
"""
import numpy as np

def compute_barth_jespersen_limiter(grid, q, grad_q):
    """
    Calculates the limiter Phi [0..1] for every cell.
    Phi = min(1.0, alpha_neighbors)
    
    Args:
        grid: The Grid object
        q: State vector [N_cells, N_vars]
        grad_q: Gradients [N_cells, N_vars, 2]
    
    Returns:
        phi: Limiter values [N_cells, N_vars]
    """
    n_cells = grid.n_cells
    n_vars = q.shape[1]
    
    # 1. Find Min/Max Q in the neighborhood of every cell
    # We initialize min/max with the cell's own value
    q_min = q.copy()
    q_max = q.copy()
    
    # Loop over faces to check neighbors (this effectively loops over the stencil)
    for i in range(grid.n_faces):
        c_L = grid.face_cells[i, 0]
        c_R = grid.face_cells[i, 1]
        
        if c_R == -1: continue # Skip boundaries for min/max check
        
        # Update L with R's value
        q_min[c_L] = np.minimum(q_min[c_L], q[c_R])
        q_max[c_L] = np.maximum(q_max[c_L], q[c_R])
        
        # Update R with L's value
        q_min[c_R] = np.minimum(q_min[c_R], q[c_L])
        q_max[c_R] = np.maximum(q_max[c_R], q[c_L])
        
    # 2. Reconstruct Q at all vertices of the cell (approximate via faces)
    # Actually, for BJ, we limit based on the reconstruction at the FACE centers.
    # Q_face = Q_c + GradQ . r
    # We want Q_face to be between Q_min and Q_max.
    
    # We compute a limiter 'phi' for each cell such that:
    # Q_face_limited = Q_c + phi * (GradQ . r) is within bounds.
    
    phi = np.ones((n_cells, n_vars), dtype=float)
    
    # Small number to prevent divide by zero
    epsilon = 1e-12
    
    for i in range(grid.n_faces):
        # We check both Left and Right sides of the face
        for side in [0, 1]:
            c_curr = grid.face_cells[i, side]
            if c_curr == -1: continue
            
            # Vector from Cell Center to Face Center
            dx = grid.face_midpoints[i, 0] - grid.cell_centers[c_curr, 0]
            dy = grid.face_midpoints[i, 1] - grid.cell_centers[c_curr, 1]
            
            # Unlimted Delta: dQ = GradQ . dr
            dq = (grad_q[c_curr, :, 0] * dx) + (grad_q[c_curr, :, 1] * dy)
            
            # Check for violation
            # If dq > 0, we risk exceeding q_max
            # If dq < 0, we risk going under q_min
            
            # Vectorized check for all variables
            for v in range(n_vars):
                p = phi[c_curr, v] # Current limiter
                
                if dq[v] > 0:
                    # Max allowed deviation
                    dq_max = q_max[c_curr, v] - q[c_curr, v]
                    if dq[v] > dq_max:
                        p = min(p, dq_max / (dq[v] + epsilon))
                elif dq[v] < 0:
                    # Min allowed deviation (negative)
                    dq_min = q_min[c_curr, v] - q[c_curr, v]
                    if dq[v] < dq_min:
                        p = min(p, dq_min / (dq[v] - epsilon))
                        
                phi[c_curr, v] = p
                
    return phi