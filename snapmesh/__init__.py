# snapmesh/__init__.py

__version__ = "1.0.0"

# Import Primitives
from .elements import Node, Edge, Cell

# Import Geometry
from .geometry import GeometricConstraint, LineSegment, Circle, Arc, PolyLine
# Import Mesh Container
from .mesh import Mesh

# Import Quality Tools
from .quality import MeshQuality  

# Import Refinement (if you still have it)
# from .refine import refine_global