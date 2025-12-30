"""
ex02b_manual_topology.py
------------------------
Goal: Manually link Nodes -> Edges -> Cell.
This demonstrates the "Strict Topology" required by the elements.py module.
"""
from snapmesh.elements import Node, Edge, Cell

def run_topology_demo():
    print("--- 1. Creating Nodes ---")
    # Define 3 nodes (Counter-Clockwise order roughly)
    n1 = Node(1, 0.0, 0.0)
    n2 = Node(2, 1.0, 0.0)
    n3 = Node(3, 0.0, 1.0)
    
    print(f"Created: {n1}")
    print(f"Created: {n2}")
    print(f"Created: {n3}")
    
    print("\n--- 2. Creating Edges ---")
    # Link the nodes with Edges.
    # Note: Edge IDs (101, 102, 103) are arbitrary here.
    e1 = Edge(101, n1, n2)  # Bottom
    e2 = Edge(102, n2, n3)  # Hypotenuse
    e3 = Edge(103, n3, n1)  # Vertical left
    
    print(f"Created: {e1}")
    print(f"   -> Normal Vector: {e1.normal}") 
    print(f"Created: {e2}")
    print(f"Created: {e3}")

    print("\n--- 3. Creating Cell ---")
    # The Cell constructor checks the winding order automatically.
    # If we passed them in the wrong order, it would swap them internally.
    c1 = Cell(1, n1, n2, n3, e1, e2, e3)
    
    print(f"Created: {c1}")
    print(f"   -> Area:   {c1.area:.4f}")
    print(f"   -> Center: {c1.center}")
    
    # Verify the cell stored the nodes correctly (it might have swapped them if we were sloppy!)
    print(f"   -> Nodes stored: {c1.n1.id}, {c1.n2.id}, {c1.n3.id}")

if __name__ == "__main__":
    run_topology_demo()