import numpy as np
from scipy.spatial import Delaunay
import matplotlib.path as mpath
import snapmesh as sm 

def generate_unstructured_mesh(boundary_curve, sizing_function, h_base=0.01, n_smooth=10):
    """
    Generates an unstructured triangular mesh inside a boundary.
    Includes Lloyd's Relaxation for high-quality triangles.
    """
    
    # 1. Bounding Box & Background Grid
    poly = np.array(boundary_curve)
    x_min, x_max = poly[:,0].min(), poly[:,0].max()
    y_min, y_max = poly[:,1].min(), poly[:,1].max()
    
    h_min = h_base / 2.0 
    xs = np.arange(x_min, x_max, h_min)
    ys = np.arange(y_min, y_max, h_min)
    xx, yy = np.meshgrid(xs, ys)
    points_candidate = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # 2. Rejection Sampling (Sizing Function)
    path = mpath.Path(poly)
    mask_inside = path.contains_points(points_candidate)
    points_inside = points_candidate[mask_inside]
    
    target_h = sizing_function(points_inside[:,0], points_inside[:,1])
    probability = np.clip((h_min / target_h)**2, 0.0, 1.0)
    
    mask_keep = np.random.random(len(points_inside)) < probability
    internal_points = points_inside[mask_keep]
    
    # 3. Fixed Boundary Points
    boundary_points = []
    for i in range(len(poly)):
        p1 = poly[i]; p2 = poly[(i+1) % len(poly)]
        seg_len = np.linalg.norm(p2 - p1)
        mid_x, mid_y = 0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])
        local_h = sizing_function(np.array([mid_x]), np.array([mid_y]))[0]
        n_seg = max(1, int(seg_len / local_h))
        for k in range(n_seg):
            t = k / n_seg
            boundary_points.append(p1 * (1-t) + p2 * t)
            
    boundary_points = np.array(boundary_points)
    
    # Combine
    if len(internal_points) > 0:
        points = np.vstack([boundary_points, internal_points])
    else:
        points = boundary_points
    
    # 4. Lloyd's Relaxation (Smoothing)
    # We move each internal point to the centroid of its Voronoi region.
    # We approximate this by the centroid of connected triangles.
    
    n_fixed = len(boundary_points)
    
    for i in range(n_smooth):
        tri = Delaunay(points)
        
        # Filter outside triangles first (concave hulls check)
        centroids = np.mean(points[tri.simplices], axis=1)
        mask_tri_inside = path.contains_points(centroids)
        simplices = tri.simplices[mask_tri_inside]
        
        # Calculate Valence (Connectivity) and Sum of Neighbors
        # A simple, fast Laplacian smooth: Point_New = Average(Neighbors)
        
        # Accumulators
        move_sum = np.zeros_like(points)
        move_count = np.zeros(len(points))
        
        # Vectorized accumulation over simplices
        # Triangle nodes: A, B, C
        A = simplices[:,0]; B = simplices[:,1]; C = simplices[:,2]
        
        # A connects to B and C
        # B connects to A and C
        # C connects to A and B
        
        # Add positions to accumulators
        np.add.at(move_sum, A, points[B] + points[C])
        np.add.at(move_count, A, 2)
        
        np.add.at(move_sum, B, points[A] + points[C])
        np.add.at(move_count, B, 2)
        
        np.add.at(move_sum, C, points[A] + points[B])
        np.add.at(move_count, C, 2)
        
        # Update Internal Points Only
        # New = Sum / Count
        mask_internal = move_count > 0
        mask_internal[:n_fixed] = False # Don't move boundary
        
        points[mask_internal] = move_sum[mask_internal] / move_count[mask_internal][:, None]

    # Final Triangulation
    tri = Delaunay(points)
    centroids = np.mean(points[tri.simplices], axis=1)
    mask_tri_inside = path.contains_points(centroids)
    simplices = tri.simplices[mask_tri_inside]
    
    return points, simplices

def raw_to_snapmesh(points, triangles):
    """
    Converts raw points/triangles to a solver-ready Mesh object.
    Auto-detects boundary tags based on position.
    """
    mesh = sm.Mesh()
    idx_to_id = {}
    
    # Add Nodes
    for i, (x, y) in enumerate(points):
        n = mesh.add_node(x, y)
        idx_to_id[i] = n.id
        
    # Add Cells
    for tri in triangles:
        mesh.add_cell(idx_to_id[tri[0]], idx_to_id[tri[1]], idx_to_id[tri[2]])
        
    # Auto-Tag Boundaries (Edge counts)
    edge_counts = {}
    for tri in triangles:
        es = [tuple(sorted((tri[0], tri[1]))), 
              tuple(sorted((tri[1], tri[2]))), 
              tuple(sorted((tri[2], tri[0])))]
        for e in es:
            edge_counts[e] = edge_counts.get(e, 0) + 1
            
    # Tag exposed edges
    for edge, count in edge_counts.items():
        if count == 1:
            p1 = points[edge[0]]; p2 = points[edge[1]]
            mx, my = 0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])
            
            # GEOMETRIC TAGGING
            if mx < 0.012:      tag = "Left"   # Inlet
            elif mx > 0.098:    tag = "Right"  # Outlet
            elif my < 0.001:    tag = "Bottom" # Centerline
            else:               tag = "Top"    # Wall
            
            mesh.tag_boundary_edge(idx_to_id[edge[0]], idx_to_id[edge[1]], tag)
            
    return mesh