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
* **`snapfvm.solver`**: The physics engine. A strictly typed Finite Volume solver supporting generic physics models.
* **`snapfvm.display`**: The reporting tool. Provides standardized, clean console output for simulations.

## Examples Guide

The `examples/` directory contains a progression from basic Python setup to full CFD simulation. Files are numbered by category.

### 01. Preliminaries (`ex01_...`)
* **Goal:** Verify the environment and understand basic tools.
* `ex01a_imports.py`: Verifies that `snapmesh` and `snapfvm` are installed.
* `ex01b_vector_math.py`: Demonstrates NumPy vector operations (norms, cross products).
* `ex01c_basic_plotting.py`: Standard boilerplate for visualizing points/lines with Matplotlib.

### 02. Core Topology (`ex02_...`)
* **Goal:** Understanding the low-level data structures (`Node`, `Cell`).
* `ex02a_create_nodes.py`: Instantiating strict `Node` objects.
* `ex02b_manual_topology.py`: Manually linking Nodes -> Edges -> Cells to build a graph.
* `ex02c_winding_check.py`: Demonstrates auto-correction of winding order (CCW).

### 03. Geometry Definition (`ex03_...`)
* **Goal:** Defining analytical domain boundaries (Pre-Meshing).
* `ex03a_geometry_objects.py`: Using `LineSegment` and `Arc` primitives.
* `ex03b_parametric_nozzle.py`: Defining a complex rocket nozzle shape without meshing it.

### 04. Structured & Mapped Meshing (`ex04_...`)
* **Goal:** Grids that follow logical (i, j) ordering.
* `ex04a_cartesian_grid.py`: Generating a simple orthogonal grid.
* `ex04b_mapped_mesh.py`: Transfinite Interpolation for 4-sided blocks.
* `ex04c_multiblock_stitch.py`: Stitching multiple mapped blocks together.

### 05. Unstructured Meshing (`ex05_...`)
* **Goal:** Delaunay triangulation and adaptation for complex shapes.
* `ex05a_simple_circle.py`: Triangulation of a basic convex shape.
* `ex05b_internal_holes.py`: Meshing with internal boundaries (Boolean subtraction).
* `ex05c_adaptive_nozzle.py`: Physics-based refinement (Fine throat, Coarse inlet).
* `ex05d_quality_check.py`: Inspecting mesh health (Aspect Ratio histograms).

### 06. Solver Fundamentals (`ex06_...`)
* **Goal:** Bridging the gap between Mesh and Physics.
* `ex06a_build_grid.py`: Converting `Mesh` topology to solver-ready `Grid` arrays.
* `ex06b_connectivity.py`: Visualizing neighbor connections (Face-to-Cell).
* `ex06c_flux_kernels.py`: Unit testing the numerical flux functions (Numba).

### 07. Compressible Flow Applications (`ex07_...`)
* **Goal:** Full CFD Simulations.
* `ex07a_supersonic_bump.py`: Mach 1.5 flow over a circular arc (Shock capturing).
* `ex07b_rocket_nozzle.py`: Choked flow simulation in a DeLaval nozzle.
* `ex07c_nozzle_validation.py`: Comparing CFD results against 1D Isentropic Theory.