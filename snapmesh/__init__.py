# snapmesh/__init__.py

__version__ = "1.0"

# Import Primitives
from .elements import Node, Edge, Cell

# Import Geometry
from .geometry import LineSegment, Circle

#from .geometry import LineSegment, Point, Arc, Circle, Line, CompositeCurve, CubicCurve, Polygon, RegularPolygon


# Import the Mesh class 
from .mesh import Mesh

from .refine import refine_global