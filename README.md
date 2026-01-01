# SnapMesh & SnapFVM

A lightweight, Python-native unstructured meshing and Finite Volume Method (FVM) solver library. 

This project is built from the ground up to be readable and educational. It avoids "black box" C++ backends in favor of clear, strictly typed Python "Smart Objects" (`Node`, `Edge`, `Cell`) that handle their own topology and geometry.

## Core Modules

* **`snapmesh.elements`**: The building blocks. Defines `Node` (vertices), `Edge` (connectivity/faces), and `Cell` (control volumes). Strictly enforces counter-clockwise (CCW) winding and checks for geometric degeneracy.
* **`snapmesh.mesh`**: The container. Manages collections of elements and handles raw-to-object conversion.
* **`snapmesh.geometry`**: Object-oriented primitives (`LineSegment`, `Arc`) for defining precise boundaries and adaptive discretization.
* **`snapmesh.unstructured_gen`**: The generator. Uses Frontal-Delaunay logic to build high-quality triangular meshes from polygon boundaries.
* **`snapmesh.quality`**: The inspector. Tools for checking mesh health (Aspect Ratio, Skewness, Area).
* **`snapmesh.grid`**: The solver connectivity. Converts a topological Mesh into a computational Grid.

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
* `ex02b_manual_topology.py`: Manually linking Nodes -> Edges -> Cells to understand the connectivity graph.
* *(Planned)* `ex02c_winding_check.py`: Demonstrates the `Cell` class auto-correcting winding order (CCW).

### 03. Mesh Operations & Generation (`ex03_...`)
* **Goal:** Generating, managing, and inspecting meshes.
* `ex03a_structured_grid.py`: Generating a simple orthogonal structured grid.
* `ex03b_quality_check.py`: Using `MeshQuality` to calculate metrics (Area, Aspect Ratio) and visualize histograms.
* `ex03c_mapped_mesh.py`: Transfinite Interpolation (Mapped Meshing) for 4-sided blocks.
* `ex03d_multiblock_nozzle.py`: Stitching multiple mapped blocks together with Node Merging.
* `ex03e_unstructured_circle.py`: Unstructured triangulation of a circle (Convex Hull test).
* `ex03f_square_hole.py`: Unstructured meshing with internal boundaries (Holes) using Boolean logic.
* `ex03g_unstructured_nozzle.py`: Basic unstructured meshing of the parametric Nozzle.
* `ex03h_refined_nozzle.py`: Adaptive unstructured meshing with physics-based local refinement (Fine Throat, Coarse Inlet).

### 04. Geometry & Constraints (`ex04_...`)
* **Goal:** Defining ideal shapes for mesh generation.
* `ex04a_geometry.py`: Creating basic Constraint objects.
* *(Planned)* `ex04b_snapping.py`: Using `node.snap()` to project vertices onto boundaries.
* `ex04c_geometry_objects.py`: Using high-level `LineSegment` and `Arc` objects to define and discretize complex boundaries (Nozzle).

### 05. Solver Setup (`ex05_...`)
* **Goal:** Preparing the mesh for physics.
* *(Planned)* `ex05a_build_grid.py`: Converting `Mesh` -> `UnstructuredGrid`.
* *(Planned)* `ex05b_connectivity.py`: Verifying neighbor connectivity (Face-to-Cell).

### 06. Simulation (`ex06_...`)
* **Goal:** Running Finite Volume simulations.
* *(Planned)* `ex06a_diffusion.py`: Steady-state diffusion.