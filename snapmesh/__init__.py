# snapmesh/__init__.py

__version__ = "1.0"

# Import Primitives
from .elements import Node, Edge, Cell

# Import the Mesh class 
from .mesh import Mesh

# Import Geometry
from .geometry import Point, Arc, Circle, Line, CompositeCurve, CubicCurve, Polygon, RegularPolygon
from .refine import refine_global