# Snapmesh

**Snapmesh** is a geometry-aware adaptive meshing tool.

## Features
- **Constraint-Based Meshing:** Define geometry (Circles, Lines) first, and the mesh snaps to it.
- **Adaptive Refinement:** Automatically refines specific regions (like wedges or boundary layers) without creating hanging nodes.
- **Pure Python:** Lightweight and easy to extend.

## Installation

```bash
pip install snapmesh

# Snapmesh

... (introduction text) ...

## Quick Start

```python
import snapmesh.mesh as sm

# Define a circular wall
wall = sm.CircleConstraint(0, 0, 10.0)

# Create a mesh
m = sm.Mesh()

# Add nodes (simplified for brevity)
# ...
