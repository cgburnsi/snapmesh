"""
snapmesh/unstructured_gen.py
----------------------------
Generates unstructured triangular meshes.
UPDATED: Implements Adaptive Topology (Splitting) to fix density mismatches.
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import matplotlib.path as mpath
from snapmesh.mesh import Mesh

def generate_unstructured_mesh(mesh_obj, sizing_func, h_base=0.1, n_smooth=80):
    print(f"--- Unstructured Gen (Adaptive DistMesh) ---")
    
    # 1. Extract Boundary
    boundary_nodes = list(mesh_obj.nodes.values())
    pts_boundary = np.array([[n.x, n.y] for n in boundary_nodes])
    
    # Identify Anchors (Corners)
    locked_mask = np.array([getattr(n, 'is_corner', False) for n in boundary_nodes], dtype=bool)
    print(f"   -> Locked {np.sum(locked_mask)} corner nodes.")
    
    path_outer = mpath.Path(pts_boundary)
    
    # 2. Initial Interior Cloud
    x_min, x_max = pts_boundary[:,0].min(), pts_boundary[:,0].max()
    y_min, y_max = pts_boundary[:,1].min(), pts_boundary[:,1].max()
    
    detected_h = [sizing_func(p[0], p[1]) for p in pts_boundary]
    # Start slightly denser to ensure coverage, let the springs handle spacing
    h_grid = min(h_base, np.min(detected_h) * 0.75 if detected_h else h_base)
    
    xs = np.arange(x_min, x_max, h_grid)
    ys = np.arange(y_min, y_max, h_grid)
    xx, yy = np.meshgrid(xs, ys)
    xx[::2] += h_grid * 0.5 # Hex packing
    
    candidates = np.vstack([xx.ravel(), yy.ravel()]).T
    mask_keep = path_outer.contains_points(candidates)
    candidates = candidates[mask_keep]
    
    # Filter candidates
    if len(candidates) > 0:
        # Probability filter
        target_h = sizing_func(candidates[:,0], candidates[:,1])
        target_h = np.maximum(np.atleast_1d(target_h), 1e-9)
        prob = np.clip((h_grid / target_h)**2, 0.0, 1.0)
        candidates = candidates[np.random.random(len(candidates)) < prob]
        
        # Wall proximity filter
        tree = cKDTree(pts_boundary)
        dists, _ = tree.query(candidates)
        candidates = candidates[dists > 0.6 * sizing_func(candidates[:,0], candidates[:,1])]

    all_points = np.vstack([pts_boundary, candidates])
    n_fixed = len(pts_boundary)
    
    full_locked_mask = np.zeros(len(all_points), dtype=bool)
    full_locked_mask[:n_fixed] = locked_mask
    
    dt = 0.2
    
    # --- HELPER: Topology Mutation ---
    def adapt_topology(points, edges, sizing_f):
        # Calculate Edge Lengths
        p0 = points[edges[:,0]]
        p1 = points[edges[:,1]]
        lengths = np.linalg.norm(p1 - p0, axis=1)
        
        # Get Target Lengths
        mid = (p0 + p1) / 2.0
        targets = sizing_f(mid[:,0], mid[:,1])
        targets = np.maximum(np.atleast_1d(targets), 1e-9)
        
        # CRITERIA: Split if Length > 1.5 * Target
        # (This fills gaps where triangles are stretched)
        split_mask = lengths > 1.5 * targets
        
        new_nodes = []
        if np.any(split_mask):
            # Add midpoints
            new_nodes = mid[split_mask]
            
        return new_nodes

    print(f"   -> Smoothing & Adapting ({n_smooth} iterations)...")
    
    for i in range(n_smooth):
        # A. Triangulate
        tri = Delaunay(all_points)
        centers = np.mean(all_points[tri.simplices], axis=1)
        mask_good = path_outer.contains_points(centers)
        good_simps = tri.simplices[mask_good]
        
        # B. Unique Edges
        edges = np.vstack([
            good_simps[:, [0,1]],
            good_simps[:, [1,2]],
            good_simps[:, [2,0]]
        ])
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        
        # --- ADAPTIVE STEP (Every 10 iters) ---
        # We inject new nodes into the array if springs are over-stretched
        if i > 0 and i % 10 == 0 and i < (n_smooth - 10):
            new_pts = adapt_topology(all_points, edges, sizing_func)
            if len(new_pts) > 0:
                # print(f"      Iter {i}: Injected {len(new_pts)} nodes to fix gaps.")
                # Append new points
                all_points = np.vstack([all_points, new_pts])
                # Update locked mask (new points are interior, so False)
                full_locked_mask = np.append(full_locked_mask, np.zeros(len(new_pts), dtype=bool))
                # Skip force calc this turn, re-triangulate next turn
                continue

        # C. Spring Forces
        p0 = all_points[edges[:,0]]
        p1 = all_points[edges[:,1]]
        vec = p1 - p0
        L = np.linalg.norm(vec, axis=1)
        L = np.maximum(L, 1e-12)
        
        mid = (p0 + p1) / 2.0
        h_target = sizing_func(mid[:,0], mid[:,1])
        h_target = np.maximum(np.atleast_1d(h_target), 1e-9)
        
        # F = L - target (Linear Spring)
        force_mag = (L - h_target)
        force_vec = vec * (force_mag / L)[:, None]
        
        deltas = np.zeros_like(all_points)
        np.add.at(deltas, edges[:,0],  force_vec)
        np.add.at(deltas, edges[:,1], -force_vec)
        
        # D. Move & Snap
        mask_move = ~full_locked_mask
        all_points[mask_move] += dt * deltas[mask_move]
        
        # Enforce Boundary Constraints
        for k in range(n_fixed):
            if not full_locked_mask[k]:
                node = boundary_nodes[k]
                # Update Node Object
                node.x = all_points[k, 0]
                node.y = all_points[k, 1]
                
                # Snap to Geometry
                if node.constraint:
                    node.snap()
                
                # Write back to array
                all_points[k, 0] = node.x
                all_points[k, 1] = node.y

    # 5. Final Export
    tri = Delaunay(all_points)
    centers = np.mean(all_points[tri.simplices], axis=1)
    mask_final = path_outer.contains_points(centers)
    final_tris = tri.simplices[mask_final]
    
    # Map back to Mesh
    idx_map = {}
    for i in range(n_fixed):
        idx_map[i] = boundary_nodes[i].id
        
    for i in range(n_fixed, len(all_points)):
        n = mesh_obj.add_node(all_points[i,0], all_points[i,1])
        idx_map[i] = n.id
        
    for t in final_tris:
        mesh_obj.add_cell(idx_map[t[0]], idx_map[t[1]], idx_map[t[2]])
        
    return mesh_obj