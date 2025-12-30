"""
ex01b_vector_math.py
--------------------
Goal: Demonstrate NumPy vector operations used in snapmesh.
SnapMesh relies on numpy for performance. This script shows 
how we calculate distance and normals.
"""
import numpy as np

def run_vector_demo():
    print("--- 1. Points as Arrays ---")
    # In pure Python, a point is a list: [x, y]
    p1_list = [0.0, 0.0]
    p2_list = [3.0, 4.0]
    
    # In SnapMesh, we convert these to numpy arrays for math
    p1 = np.array(p1_list)
    p2 = np.array(p2_list)
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")

    print("\n--- 2. Vector Subtraction (The 'Edge Vector') ---")
    # Finding the vector from p1 to p2 is just subtraction
    # v = p2 - p1
    vector = p2 - p1
    print(f"Vector (p2 - p1): {vector}")  # Should be [3. 4.]

    print("\n--- 3. Distance (Edge Length) ---")
    # We use Linear Algebra Norm (Euclidean distance)
    dist = np.linalg.norm(vector)
    print(f"Length (Norm): {dist}")       # Should be 5.0

    print("\n--- 4. Normal Vectors (Rotations) ---")
    # To find a 2D normal, we rotate the vector 90 degrees.
    # Formula: (dx, dy) -> (-dy, dx)
    dx, dy = vector[0], vector[1]
    normal_unscaled = np.array([-dy, dx])
    print(f"Rotated Vector: {normal_unscaled}")
    
    # Normalize to length 1.0 (Unit Vector)
    normal_unit = normal_unscaled / np.linalg.norm(normal_unscaled)
    print(f"Unit Normal: {normal_unit}")

if __name__ == "__main__":
    run_vector_demo()