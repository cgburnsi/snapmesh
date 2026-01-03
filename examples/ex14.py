import numpy as np
import matplotlib.pyplot as plt
import snapmesh.unstructured_gen as umesh # The file above
import snapcore.units as cv

def nozzle_sizing_function(x, y):
    """
    Returns desired edge length 'h' at position (x,y).
    """
    # 1. Base Size (Core flow)
    h_background = 0.005  # 5mm
    
    # 2. Wall Distance (Approximate)
    # We roughly know the wall is at radius ~0.02 to 0.06m
    # A simple approximation for the nozzle wall:
    # Just use 'y' coordinate to bias?
    # Or measure distance to a simplified line.
    
    # Let's refine based on "High Y" (Near wall) and "Throat X"
    
    # Refine near Throat (x ~ 0.065)
    dist_throat = np.abs(x - 0.065)
    factor_throat = 0.5 + 0.5 * np.tanh(dist_throat / 0.02) # 0.5 at throat, 1.0 away
    
    # Refine near Wall (Large Y)
    # We want small h when y is large.
    # Sigmoid function of y?
    # Let's say h decreases as y increases.
    factor_wall = 1.0 - 0.7 * (y / 0.06) # At y=0 -> 1.0. At y=0.06 -> 0.3
    factor_wall = np.clip(factor_wall, 0.2, 1.0)
    
    # Combine
    h = h_background * factor_throat * factor_wall
    return h

def run_mesher():
    # 1. Define Boundary Polygon (Discretized)
    # We manually trace the "Structured" curves to get a polygon loop
    # (In a real app, we'd sample the CompositeCurve from ex8)
    
    # Simplified Nozzle Polygon for demonstration
    xi, xe = 0.01, 0.10
    ri, re = 0.06, 0.03
    rt = 0.02
    xt = (xi + xe) / 2
    
    # Create points walking CCW
    # Bottom (Inlet -> Exit)
    pts_bottom = np.column_stack([np.linspace(xi, xe, 20), np.zeros(20)])
    
    # Right (Exit)
    pts_exit = np.column_stack([np.full(10, xe), np.linspace(0, re, 10)])
    
    # Top (Wall) - Cosine approximation
    x_wall = np.linspace(xe, xi, 40) # Backwards
    # Simple nozzle shape math
    y_wall = rt + (ri - rt) * ((x_wall - xt)/(xi - xt))**2
    # Fix the exit part roughly
    mask_div = x_wall > xt
    y_wall[mask_div] = rt + (re - rt) * ((x_wall[mask_div] - xt)/(xe - xt))**2
    
    pts_top = np.column_stack([x_wall, y_wall])
    
    # Left (Inlet)
    pts_inlet = np.column_stack([np.full(10, xi), np.linspace(ri, 0, 10)])
    
    # Combine loop
    boundary = np.vstack([pts_bottom, pts_exit, pts_top, pts_inlet])
    
    # 2. Generate Mesh
    print("Generating Unstructured Mesh...")
    points, cells = umesh.generate_unstructured_mesh(boundary, nozzle_sizing_function, h_base=0.005)
    
    print(f"Generated {len(points)} nodes, {len(cells)} cells.")
    
    # 3. Plot
    plt.figure(figsize=(10,5))
    plt.triplot(points[:,0], points[:,1], cells, 'k-', lw=0.5)
    plt.plot(points[:,0], points[:,1], 'r.', markersize=2)
    plt.axis('equal')
    plt.title("Unstructured Mesh with Sizing Function")
    plt.show()

if __name__ == "__main__":
    run_mesher()