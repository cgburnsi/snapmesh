# SnapMesh & SnapFVM

A lightweight, Python-native engineering suite for unstructured meshing, Finite Volume Method (FVM) solving, and chemical kinetics.

This project is built from the ground up to be readable and educational. It avoids "black box" C++ backends in favor of clear, strictly typed Python "Smart Objects" that handle their own topology, geometry, and physics.

## The Snap Suite

The library is organized into four distinct packages, each with a specific responsibility:

### 1. `snapcore` (The Foundation)
* **Shared Utilities:** Universal tools used across the suite.
* **`snapcore.units`**: Strict unit conversion (avoiding "magic number" errors).
* **`snapcore.display`**: Standardized console reporting and logging.

### 2. `snapmesh` (The Geometer)
* **Pure Geometry:** Defines the domain without knowing about physics.
* **`snapmesh.geometry`**: Primitives like `LineSegment`, `Arc`, and `Spline`.
* **`snapmesh.unstructured_gen`**: Frontal-Delaunay generator for high-quality triangular meshes.
* **`snapmesh.topology`**: Strictly typed `Node`, `Edge`, and `Cell` objects with CCW winding enforcement.

### 3. `snapfvm` (The Solver)
* **Numerical Engine:** The Finite Volume integrator.
* **`snapfvm.grid`**: Converts topological meshes into computational grids (calculates face normals, volumes, neighbor matrices).
* **`snapfvm.solver`**: The time-stepping loop and residual monitor.
* **`snapfvm.physics`**: Modular physics definitions (e.g., `Euler2D`, `Incompressible`).

### 4. `snapchem` (The Chemist)
* **Thermodynamics & Kinetics:** Independent chemistry handling.
* **`snapchem.gas_phase`**: Equation of State solvers (Ideal Gas, Redlich-Kwong).
* **`snapchem.kinetics`**: Reaction rate calculations (Arrhenius).
* **`snapchem.surface`**: Catalyst site density and surface reaction models.

---

## Examples Guide

The `examples/` directory contains a linear progression from basic Python setup to full multi-physics simulations.

### 01. Preliminaries
* **Goal:** Verify the environment and understand basic tools.
* `ex01a_imports.py`: Verifies the library installation.
* `ex01b_vector_math.py`: Demonstrates vector operations.

### 02. Core Topology
* **Goal:** Understanding the low-level data structures.
* `ex02a_create_nodes.py`: Instantiating strict `Node` objects.
* `ex02b_manual_topology.py`: Manually linking Nodes -> Edges -> Cells.

### 03. Geometry Definition
* **Goal:** Defining analytical domain boundaries (Pre-Meshing).
* `ex03a_geometry_objects.py`: Using `LineSegment` and `Arc` primitives.
* `ex03b_parametric_nozzle.py`: Defining complex shapes.

### 04. Structured & Mapped Meshing
* **Goal:** Grids that follow logical (i, j) ordering.
* `ex04a_cartesian_grid.py`: Generating simple orthogonal grids.
* `ex04b_mapped_mesh.py`: Transfinite Interpolation.

### 05. Unstructured Meshing
* **Goal:** Delaunay triangulation and adaptation for complex shapes.
* `ex05a_simple_circle.py`: Basic convex shapes.
* `ex05c_adaptive_nozzle.py`: Physics-based refinement (Fine throat, Coarse inlet).

### 06. Solver Fundamentals
* **Goal:** Bridging the gap between Mesh and Physics.
* `ex06a_build_grid.py`: Converting `Mesh` topology to solver-ready `Grid` arrays.
* `ex06c_flux_kernels.py`: Unit testing numerical flux functions.

### 07. Incompressible Flow
* **Goal:** Solving for Pressure and Velocity (Low Speed).
* *(Planned)* `ex07a_poiseuille_channel.py`: Laminar flow in a pipe.
* *(Planned)* `ex07b_lid_driven_cavity.py`: The classic CFD benchmark.

### 08. Viscous & Turbulent Flow
* **Goal:** Adding diffusion and turbulence models.
* *(Planned)* `ex08a_viscous_cylinder.py`: Flow separation.
* *(Planned)* `ex08b_spalart_allmaras.py`: 1-equation turbulence modeling.

### 09. Compressible Flow (Fundamentals)
* **Goal:** Solving Euler Equations (High Speed, Shock Waves).
* *(Planned)* `ex09a_sod_shock_tube.py`: 1D Riemann problem validation.
* *(Planned)* `ex09b_supersonic_wedge.py`: Oblique shock capture.

### 10. Applied Aerospace
* **Goal:** Full system simulations.
* `ex10a_supersonic_bump.py`: Mach 1.5 flow over a circular arc.
* `ex10b_rocket_nozzle.py`: Choked flow simulation in a DeLaval nozzle.
* `ex10c_nozzle_validation.py`: Validation against 1D Isentropic Theory.

### 11. Scalar Transport & Thermodynamics
* **Goal:** Moving beyond pure fluid dynamics.
* *(Planned)* `ex11a_passive_scalar.py`: Convection-Diffusion of a tracer.
* *(Planned)* `ex11b_heated_plate.py`: Thermal conduction.

### 12. Reacting Flows & Porous Media
* **Goal:** Chemical Reactors and Catalyst Beds.
* *(Planned)* `ex12a_darcy_flow.py`: Flow through a porous block.
* *(Planned)* `ex12b_surface_reaction.py`: Simple A -> B reaction on a wall.
* *(Planned)* `ex12c_packed_bed_reactor.py`: Full simulation of a catalytic bed (Flow + Heat + Reaction).