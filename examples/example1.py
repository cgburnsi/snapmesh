import snapmesh.mesh as sm

# Define a circular wall
wall = sm.CircleConstraint(0, 0, 10.0)

# Create a mesh and refine it
m = sm.Mesh()
# ... (add nodes) ...
refined = sm.refine_global(m)
