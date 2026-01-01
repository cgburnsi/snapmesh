"""
snapmesh/unstructured_gen.py
----------------------------
Generates unstructured triangular meshes.
Features:
- Boolean Hole Logic (Robust)
- Variable Density Support
- Proximity Filtering (Fixes boundary slivers)
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import matplotlib.path as mpath
from snapmesh.mesh import Mesh

def generate_unstructured_mesh(boundary_parts, sizing_func, h_base=0.1, n_smooth=20):
    print(f"--- Unstructured Gen (h_base={h_base}) ---")
    
    # 0. Normalize Input
    if not isinstance(boundary_parts[0][0], (list, tuple, np.ndarray)):
        polys = [np.array(boundary_parts)]
    else:
        polys = [np.array(p) for p in boundary_parts]
        
    # 1. Analyze Geometry (Outer vs Inner)
    areas = []
    for p in polys:
        xmin, xmax = p[:,0].min(), p[:,0].max()
        ymin, ymax = p[:,1].min(), p[:,1].max()
        areas.append((xmax-xmin)*(ymax-ymin))
    
    outer_idx = np.argmax(areas)
    outer_poly = polys[outer_idx]
    inner_polys = [p for i, p in enumerate(polys) if i != outer_idx]
    
    print(f"   -> Detected 1 Outer Loop and {len(inner_polys)} Inner Holes")

    # 2. Build Paths
    path_outer = mpath.Path(outer_poly)
    paths_inner = [mpath.Path(p) for p in inner_polys]

    # 3. Discretize Boundaries
    fixed_nodes = []
    
    for poly in polys:
        n_pts = len(poly)
        for i in range(n_pts):
            p1 = poly[i]
            p2 = poly[(i+1)%n_pts]
            dist = np.linalg.norm(p2 - p1)
            mid = (p1 + p2)/2
            
            local_h = sizing_func(mid[0], mid[1])
            if local_h is None: local_h = h_base
            
            n_sub = max(1, int(np.round(dist / local_h)))
            for k in range(n_sub):
                fixed_nodes.append(p1 + (k/n_sub)*(p2-p1))
    
    fixed_nodes = np.array(fixed_nodes)
    print(f"   -> Boundary Nodes: {len(fixed_nodes)}")

    # 4. Fill Interior
    x_min, x_max = outer_poly[:,0].min(), outer_poly[:,0].max()
    y_min, y_max = outer_poly[:,1].min(), outer_poly[:,1].max()
    
    # Fine grid for rejection sampling
    h_grid = h_base * 0.4 
    xs = np.arange(x_min, x_max, h_grid)
    ys = np.arange(y_min, y_max, h_grid)
    xx, yy = np.meshgrid(xs, ys)
    
    xx += np.random.uniform(-h_grid*0.2, h_grid*0.2, xx.shape)
    yy += np.random.uniform(-h_grid*0.2, h_grid*0.2, yy.shape)
    
    candidates = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # A. Geometric Filter (Boolean)
    mask_keep = path_outer.contains_points(candidates)
    for p_in in paths_inner:
        mask_hole = p_in.contains_points(candidates, radius=-1e-9) 
        mask_keep = mask_keep & (~mask_hole)
        
    pts_geo = candidates[mask_keep]
    
    # B. Probability Filter (Density)
    if len(pts_geo) > 0:
        target_h = sizing_func(pts_geo[:,0], pts_geo[:,1])
        target_h = np.maximum(np.atleast_1d(target_h), 1e-9)
        prob = np.clip(1.3 * (h_grid / target_h)**2, 0.0, 1.0)
        
        mask_prob = np.random.random(len(pts_geo)) < prob
        interior_nodes = pts_geo[mask_prob]
        
        # --- C. PROXIMITY FILTER (The Fix) ---
        # Remove points that are too close to the boundary
        if len(interior_nodes) > 0 and len(fixed_nodes) > 0:
            tree = cKDTree(fixed_nodes)
            dist, _ = tree.query(interior_nodes)
            
            # Re-calculate local h for the survivors
            survivor_h = sizing_func(interior_nodes[:,0], interior_nodes[:,1])
            survivor_h = np.maximum(np.atleast_1d(survivor_h), 1e-9)
            
            # Rule: Must be at least 0.6 * local_h away from any wall
            mask_far = dist > (0.6 * survivor_h)
            interior_nodes = interior_nodes[mask_far]
            
    else:
        interior_nodes = np.empty((0,2))

    all_points = np.vstack([fixed_nodes, interior_nodes])
    n_fixed = len(fixed_nodes)
    
    # 5. Smoothing
    print(f"   -> Smoothing ({n_smooth} iterations)...")
    for _ in range(n_smooth):
        tri = Delaunay(all_points)
        
        centers = np.mean(all_points[tri.simplices], axis=1)
        
        # Boolean Filter for Centroids
        mask_good = path_outer.contains_points(centers)
        for p_in in paths_inner:
            mask_in = p_in.contains_points(centers, radius=-1e-9)
            mask_good = mask_good & (~mask_in)
            
        good_simplices = tri.simplices[mask_good]
        
        neigh_sum = np.zeros_like(all_points)
        neigh_cnt = np.zeros(len(all_points))
        
        A, B, C = good_simplices[:,0], good_simplices[:,1], good_simplices[:,2]
        np.add.at(neigh_sum, A, all_points[B] + all_points[C])
        np.add.at(neigh_cnt, A, 2)
        np.add.at(neigh_sum, B, all_points[A] + all_points[C])
        np.add.at(neigh_cnt, B, 2)
        np.add.at(neigh_sum, C, all_points[A] + all_points[B])
        np.add.at(neigh_cnt, C, 2)
        
        mask_move = (neigh_cnt > 0)
        mask_move[:n_fixed] = False
        
        # Relax
        avg = neigh_sum[mask_move] / neigh_cnt[mask_move][:,None]
        all_points[mask_move] = 0.7 * all_points[mask_move] + 0.3 * avg

    # 6. Final Export
    tri = Delaunay(all_points)
    centers = np.mean(all_points[tri.simplices], axis=1)
    
    mask_final = path_outer.contains_points(centers)
    for p_in in paths_inner:
        mask_in = p_in.contains_points(centers, radius=-1e-9)
        mask_final = mask_final & (~mask_in)
        
    final_tris = tri.simplices[mask_final]
    
    mesh = Mesh()
    idx_map = {}
    for i, pt in enumerate(all_points):
        n = mesh.add_node(pt[0], pt[1])
        idx_map[i] = n.id
    for t in final_tris:
        mesh.add_cell(idx_map[t[0]], idx_map[t[1]], idx_map[t[2]])
        
    return mesh