# snapmesh/__init__.py

__version__ = "0.1.0"

# Import the Mesh class from the submodule
from .mesh import Mesh
from .geometry import Arc, Circle, Line, CompositeCurve, CubicCurve, Polygon, RegularPolygon
from .refine import refine_global


