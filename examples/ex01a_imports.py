"""
ex01a_imports.py
----------------
Goal: Verify the environment and library imports.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# Try importing the new module name
try:
    import snapmesh.elements as elements
    print("SUCCESS: 'snapmesh.elements' found.")
except ImportError:
    print("ERROR: Could not import 'snapmesh'. Make sure the 'snapmesh' folder is in this directory.")
    sys.exit(1)

def run_check():
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NumPy Version:  {np.__version__}")
    
    # Check if we can access the classes
    print("\nChecking Class Availability:")
    print(f" - Node class: {elements.Node}")
    print(f" - Edge class: {elements.Edge}")
    print(f" - Cell class: {elements.Cell}")
    print("\nEnvironment is ready for SnapMesh!")

if __name__ == "__main__":
    run_check()