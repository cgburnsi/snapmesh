"""
ex04a_geometry.py
-----------------
Goal: Test the new NumPy-based LineSegment class.
Verifies both projection (staying on line) and clamping (staying on segment).
"""
import numpy as np
from snapmesh.elements import Node
from snapmesh.geometry import LineSegment, Circle

def run_geometry_test():
    print("--- 1. Creating a LineSegment Constraint ---")
    # Segment from (0,0) to (10,0)
    wall = LineSegment((0.0, 0.0), (10.0, 0.0)) 
    print(f"Created: {wall}")
    
    print("\n--- 2. Snapping Node A (Middle) ---")
    # Node at (5, 2) -> Should snap to (5, 0)
    nA = Node(1, 5.0, 2.0)
    print(f"Before Snap A: {nA}")
    wall.snap(nA)
    print(f"After Snap A:  {nA}")

    # Check 1: Did it hit the line? (y=0)
    if abs(nA.y) < 1e-9:
        print(" -> SUCCESS: Node A snapped to line (y=0).")
    else:
        print(f" -> FAILURE: Node A y-coord is {nA.y}")

    # Check 2: Did it stay at x=5? (Orthogonal projection)
    if abs(nA.x - 5.0) < 1e-9:
        print(" -> SUCCESS: Node A kept its x-position.")
    else:
        print(f" -> FAILURE: Node A shifted x to {nA.x}")

    print("\n--- 3. Snapping Node B (Off the End) ---")
    # Node at (12, 2) -> Should snap to ENDPOINT (10, 0)
    nB = Node(2, 12.0, 2.0)
    print(f"Before Snap B: {nB}")
    wall.snap(nB)
    print(f"After Snap B:  {nB}")

    # Check 3: Did it clamp to x=10?
    if abs(nB.x - 10.0) < 1e-9 and abs(nB.y) < 1e-9:
        print(" -> SUCCESS: Node B clamped to endpoint (10, 0).")
    else:
        print(f" -> FAILURE: Node B did not clamp. Loc: ({nB.x}, {nB.y})")
        
    print("\n--- 4. Snapping to Circle ---")
    c = Circle((0,0), 5.0)
    nC = Node(3, 2.0, 2.0) # Inside the circle
    print(f"Before: {nC}")
    c.snap(nC) # Should push it out to radius 5
    print(f"After:  {nC}")
    
    # Check radius
    dist = np.sqrt(nC.x**2 + nC.y**2)
    if abs(dist - 5.0) < 1e-9:
        print("SUCCESS: Node snapped to radius 5.0")

if __name__ == "__main__":
    run_geometry_test()