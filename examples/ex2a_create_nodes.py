"""
ex02a_create_nodes.py
---------------------
Goal: Instantiate Node objects and understand their attributes.
This is the "Hello World" of the snapmesh library itself.
"""
import numpy as np
from snapmesh.elements import Node

def run_node_demo():
    print("--- 1. Simple Node Creation ---")
    # Nodes are just ID + Coordinates
    n1 = Node(1, 0.5, 0.5)
    n2 = Node(2, 3.2, 4.1)
    
    print(f"Node 1: {n1}")
    print(f"Node 2: {n2}")
    
    print("\n--- 2. Accessing Data ---")
    # You can access x and y directly
    print(f"n1.x: {n1.x}")
    print(f"n1.y: {n1.y}")
    
    print("\n--- 3. Numpy Conversion ---")
    # The solver will need numpy arrays, not objects.
    # The .to_array() helper handles this cleanly.
    p1 = n1.to_array()
    print(f"As Array: {p1} (Type: {type(p1)})")
    
    print("\n--- 4. Memory Check (Optional) ---")
    # Because we use __slots__, these objects are very small.
    # You can't add random attributes to them (Safety Feature).
    try:
        n1.color = "red" 
    except AttributeError as e:
        print(f"Safety Check Passed: Cannot add random attributes.\nError: {e}")

if __name__ == "__main__":
    run_node_demo()