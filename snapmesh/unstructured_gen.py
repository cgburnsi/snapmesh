"""
snapmesh/unstructured_gen.py
----------------------------
Generates unstructured triangular meshes using a Distmesh-like approach.
1. Discretize Boundary
2. Fill Interior
3. Delaunay + Smoothing
4. Filter Outside Cells
"""
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.path as mpath
from snapmesh.mesh import Mesh, BCTag

def generate_unstructured_mesh(boundary_poly, sizing_func, h_base=0.1, n_smooth=20):
    """
    Args:
        boundary_poly: List of (x,y) tuples defining the closed loop.
        sizing_func: Function f(x,y) -> target_edge_length.
        h_base: Baseline size if sizing_func returns None.
    """
    print(f"--- Unstructured Gen (h_base={h_base}) ---")
    
    poly = np.array(boundary_poly)
    path = mpath.Path(poly)
    
    # 1. Discretize Boundary (Fixed Nodes)
    # Walk the perimeter and place nodes according to sizing_func
    fixed_nodes = []
    
    num_seg = len(poly)
    for i in range(num_seg):
        p_start = poly[i]
        p_end   = poly[(i+1)%num_seg]
        
        seg_vec = p_end - p_start
        seg_len = np.linalg.norm(seg_vec)
        
        # Check sizing at midpoint
        mid = (p_start + p_end)/2
        local_h = sizing_func(mid[0], mid[1])
        if local_h is None: local_h = h_base
            
        n_sub = max(1, int(np.round(seg_len / local_h)))
        
        for k in range(n_sub):
            t = k / n_sub
            pos = p_start + t * seg_vec
            fixed_nodes.append(pos)
            
    fixed_nodes = np.array(fixed_nodes)
    print(f"   -> Boundary: {len(fixed_nodes)} nodes")
    
    # 2. Fill Interior (Rejection Sampling)
    x_min, x_max = poly[:,0].min(), poly[:,0].max()
    y_min, y_max = poly[:,1].min(), poly[:,1].max()
    
    # Grid slightly tighter than h_base to ensure coverage
    h_grid = h_base * 0.8
    xs = np.arange(x_min, x_max, h_grid)
    ys = np.arange(y_min, y_max, h_grid)
    xx, yy = np.meshgrid(xs, ys)
    
    # Jitter to break alignment
    xx += np.random.uniform(-h_grid*0.2, h_grid*0.2, xx.shape)
    yy += np.random.uniform(-h_grid*0.2, h_grid*0.2, yy.shape)
    
    candidates = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Keep only those inside
    mask = path.contains_points(candidates)
    interior_nodes = candidates[mask]
    
    # Density Filter (Optional: Randomly kill nodes if local_h is large)
    # For now, we assume uniform density for stability
    
    # Combine
    all_points = np.vstack([fixed_nodes, interior_nodes])
    
    # 3. Smoothing (Lloyd's Relaxation)
    # We move points to the centroid of their Voronoi cell
    # BUT we lock the 'fixed_nodes' in place.
    n_fixed = len(fixed_nodes)
    
    print(f"   -> Smoothing ({n_smooth} iterations)...")
    for _ in range(n_smooth):
        tri = Delaunay(all_points)
        
        # Calculate neighbor averages (Vectorized)
        neigh_sum = np.zeros_like(all_points)
        neigh_cnt = np.zeros(len(all_points))
        
        # Add edges: A-B, B-C, C-A
        simplices = tri.simplices
        
        # Filter out triangles that are effectively "outside" or spanning concave gaps
        # Centroid check
        centers = np.mean(all_points[simplices], axis=1)
        mask_good = path.contains_points(centers)
        good_simplices = simplices[mask_good]
        
        # Accumulate forces
        A = good_simplices[:,0]
        B = good_simplices[:,1]
        C = good_simplices[:,2]
        
        np.add.at(neigh_sum, A, all_points[B] + all_points[C])
        np.add.at(neigh_cnt, A, 2)
        
        np.add.at(neigh_sum, B, all_points[A] + all_points[C])
        np.add.at(neigh_cnt, B, 2)
        
        np.add.at(neigh_sum, C, all_points[A] + all_points[B])
        np.add.at(neigh_cnt, C, 2)
        
        # Update only interior nodes
        # P_new = P_old + omega * (Average - P_old)
        mask_move = (neigh_cnt > 0)
        mask_move[:n_fixed] = False # Lock boundary
        
        avg_pos = neigh_sum[mask_move] / neigh_cnt[mask_move][:,None]
        all_points[mask_move] = 0.6 * all_points[mask_move] + 0.4 * avg_pos
        
    # 4. Final Triangulation & Convert to SnapMesh
    tri = Delaunay(all_points)
    
    # Final cleanup of outside triangles
    centers = np.mean(all_points[tri.simplices], axis=1)
    mask_final = path.contains_points(centers)
    final_tris = tri.simplices[mask_final]
    
    # Create SnapMesh Object
    mesh = Mesh()
    idx_map = {}
    
    # Add Nodes
    for i, pt in enumerate(all_points):
        n = mesh.add_node(pt[0], pt[1])
        idx_map[i] = n.id
        
    # Add Cells
    for t in final_tris:
        mesh.add_cell(idx_map[t[0]], idx_map[t[1]], idx_map[t[2]])
        
    return mesh