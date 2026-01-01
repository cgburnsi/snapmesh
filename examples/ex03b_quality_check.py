"""
ex03b_quality_tool.py
---------------------
Goal: Test the new 'snapmesh.quality' module on a perfect grid.
"""
from snapmesh.mesh import Mesh
from snapmesh.quality import MeshQuality

# Import the grid generator from the previous step
from ex03a_structured_grid import create_structured_grid

def run():
    print("1. Generating 10x10 Grid...")
    mesh = create_structured_grid(1.0, 1.0, 10, 10)
    
    print("2. Running Quality Inspector...")
    inspector = MeshQuality(mesh)
    
    # Run analysis
    inspector.analyze()
    
    # Print text report
    inspector.print_report()
    
    # Show plots
    print("3. Plotting...")
    inspector.plot_histograms()

if __name__ == "__main__":
    run()