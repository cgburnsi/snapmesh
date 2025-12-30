# snapmesh/__init__.py

__version__ = "1.0"

# Import the Mesh class from the submodule
from .mesh import Mesh
from .geometry import Point, Arc, Circle, Line, CompositeCurve, CubicCurve, Polygon, RegularPolygon
from .refine import refine_global


