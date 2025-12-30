# SnapMesh & SnapFVM

A lightweight, Python-native unstructured meshing and Finite Volume Method (FVM) solver library. 

This project is built from the ground up to be readable and educational. It avoids "black box" C++ backends in favor of clear, strictly typed Python "Smart Objects" (`Node`, `Edge`, `Cell`) that handle their own topology and geometry.

## Core Modules

* **`snapmesh.elements`**: The building blocks. Defines `Node` (vertices), `Edge` (connectivity/faces), and `Cell` (control volumes). Strictly enforces counter-clockwise (CCW) winding and checks for geometric degeneracy.
* **`snapmesh.mesh`**: The container. Manages collections of elements and handles raw-to-object conversion.
* **`snapmesh.grid`**: The solver connectivity. Converts a topological Mesh into a computational Grid, identifying neighbors and boundary conditions for FVM.
* **`snapfvm.solver`**: (In Progress) The finite volume solver for conservation laws.

## Examples Guide

The `examples/` directory contains a step-by-step progression from basic Python setup to full CFD simulation. Files are numbered by category.

### 01. Foundations (`ex01_...`)
* **Goal:** Verify the environment and understand the underlying math/tools.
* `ex01a_imports.py`: Verifies that `snapmesh` is installed and accessible.
* `ex01b_vector_math.py`: Demonstrates the NumPy vector operations (norms, cross products) used by the library.
* `ex01c_basic_plotting.py`: Standard boilerplate for visualizing points and lines using Matplotlib.

### 02. Elements & Topology (`ex02_...`)
* **Goal:** Working with the "Smart Objects" in `elements.py`.
* `ex02a_create_nodes.py`: Instantiating strict `Node` objects and accessing their data.
* `ex02b_manual_topology.py`: Manually linking Nodes $\to$ Edges $\to$ Cells to understand the connectivity graph.
* *(Planned)* `ex02c_winding_check.py`: Demonstrates the `Cell` class auto-correcting winding order (CCW).

### 03. Mesh Operations (`ex03_...`)
* **Goal:** Managing collections of elements.
* *(Planned)* `ex03a_mesh_container.py`: Registering elements into the main `Mesh` object.
* *(Planned)* `ex03b_raw_import.py`: Converting raw point/triangle lists into a full `snapmesh`.

### 04. Geometry & Constraints (`ex04_...`)
* **Goal:** Defining ideal shapes for mesh generation.
* *(Planned)* `ex04a_geometry.py`: Creating `Line` and `Circle` constraint objects.
* *(Planned)* `ex04b_snapping.py`: Using `node.snap()` to project vertices onto boundaries.

### 05. Solver Setup (`ex05_...`)
* **Goal:** Preparing the mesh for physics.
* *(Planned)* `ex05a_build_grid.py`: converting `Mesh` $\to$ `UnstructuredGrid`.
* *(Planned)* `ex05b_connectivity.py`: Verifying neighbor connectivity (Face-to-Cell).

### 06. Simulation (`ex06_...`)
* **Goal:** Running Finite Volume simulations.
* *(Planned)* `ex06a_diffusion.py`: Steady-state diffusion (formerly `ex15.py`).

## Key Features

* **Topological Safety:** The `Cell` class automatically detects degenerate (zero-area) triangles and enforces CCW winding.
* **Face-Based Connectivity:** Edges store `[Owner, Neighbor]` references, enabling standard FVM flux loops.
* **Strict Typing:** Uses `__slots__` and strict object references to prevent "loose integer" bugs common in mesh codes.