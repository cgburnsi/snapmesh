"""
ex01c_basic_plotting.py
-----------------------
Goal: Standard boilerplate for visualizing points and lines.
Meshing requires constant visual feedback.
"""
import matplotlib.pyplot as plt

def run_plot_demo():
    # 1. Define some raw data (Square)
    x_coords = [0, 1, 1, 0, 0]  # Note: Repeat first point to close loop
    y_coords = [0, 0, 1, 1, 0]

    print("Plotting a simple square...")

    # 2. Setup the Plot
    plt.figure(figsize=(6, 6))
    
    # 3. Plot Lines (The "Edges")
    # 'b-' means Blue Line, 'o' means markers at nodes
    plt.plot(x_coords, y_coords, 'b-o', label='Boundary', linewidth=2)

    # 4. Annotate Points (Optional but helpful for debugging IDs)
    for i, (x, y) in enumerate(zip(x_coords[:-1], y_coords[:-1])):
        plt.text(x, y, f" N{i+1}", fontsize=12, color='red')

    # 5. Formatting
    plt.title("Basic Mesh Visualization")
    plt.xlabel("X Coordinate [m]")
    plt.ylabel("Y Coordinate [m]")
    plt.grid(True)
    plt.axis('equal')  # CRITICAL: Ensures the square doesn't look like a rectangle
    plt.legend()
    
    # 6. Show
    plt.show()

if __name__ == "__main__":
    run_plot_demo()