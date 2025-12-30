''' geometry.py
    -----------
    Defines geometric constraints (Lines, Circles, etc.) that Nodes can 'snap' to.
    These classes are used by the Mesh manager to enforce shape fidelity.
'''
import numpy as np
from abc import ABC, abstractmethod

import math

class Geometry(ABC):
    ''' Abstract Base Class for all geometric constraints.
        Enforces that every subclass MUST implement the 
        following methods.
    '''
    
    @abstractmethod
    def snap(self, node):
        ''' Modifies the node's (x, y) position in-place. '''
        pass


class ParametricCurve(ABC):
    ''' Abstract base for all 1D Curves. '''
    
    @abstractmethod
    def evaluate(self, t):
        ''' Returns (x, y) coordinates at parameter t (0.0 <= t <= 1.0). '''
        pass



class LineSegment(ParametricCurve):
    """ 
    A straight line connecting two points (p1, p2).
    Nodes snap to the segment. If they project outside the endpoints,
    they are clamped to the nearest endpoint (p1 or p2).
    """
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        
        self.vec = self.p2 - self.p1
        self.len_sq = np.dot(self.vec, self.vec)
        
        if self.len_sq == 0:
            raise ValueError("LineSegment cannot be zero length.")

    def evaluate(self, t):
        """ Returns point at t (0.0 = p1, 1.0 = p2). """
        return self.p1 + t * self.vec

    def snap(self, node):
        """ Projects node onto the SEGMENT. Clamps to endpoints 
            if the projection falls outside.
        """
        p = node.to_array()
        ap = p - self.p1
        
        # Projection factor t
        t = np.dot(ap, self.vec) / self.len_sq
        
        # --- CLAMPING (The "Segment" behavior) ---
        t = np.clip(t, 0.0, 1.0)
        
        # Calculate closest point
        closest = self.p1 + t * self.vec
        node.update_from_array(closest)

    def __repr__(self):
        return f"LineSegment(p1={self.p1}, p2={self.p2})"

    


class Circle(ParametricCurve):
    """
    A full circle defined by center (cx, cy) and radius r.
    Parameter t (0..1) maps to 0..2*pi radians.
    """
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float64)
        self.r = float(radius)
        
        if self.r <= 0:
            raise ValueError("Circle radius must be positive.")

    def evaluate(self, t):
        """ Returns point at parameter t (0.0 to 1.0). """
        # Map t to angle (radians)
        theta = t * 2.0 * np.pi
        
        x = self.center[0] + self.r * np.cos(theta)
        y = self.center[1] + self.r * np.sin(theta)
        return np.array([x, y])

    def snap(self, node):
        """ 
        Projects node onto the circle circumference.
        """
        p = node.to_array()
        vec = p - self.center
        
        # Calculate distance from center
        dist = np.linalg.norm(vec)
        
        # Prevent singularity at the exact center
        if dist < 1e-12:
            # Edge case: If node is at center, snap to angle 0
            node.x = self.center[0] + self.r
            node.y = self.center[1]
            return

        # Scale vector to match radius
        scale = self.r / dist
        closest = self.center + vec * scale
        
        node.update_from_array(closest)
        
    def __repr__(self):
        return f"Circle(center={self.center}, r={self.r})"
    
    
        

class Point:
    """
    Pins a node to a specific (x, y) location.
    Effectively a 'Fixed' constraint.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def snap(self, node):
        # Force the node exactly to this location
        node.x = self.x
        node.y = self.y
        
    def __repr__(self):
        return f'Point: (x, y) = {self.x:4.4f}, {self.y:4.4f}.'


'''
class Circle(ParametricCurve):
    def __init__(self, x, y, r):
        self.cx = x
        self.cy = y
        self.r = r

    def evaluate(self, t):
        # Map t=0..1 to 0..2*pi
        theta = t * 2 * math.pi
        x = self.cx + self.r * math.cos(theta)
        y = self.cy + self.r * math.sin(theta)
        return x, y

    def snap(self, node):
        dx = node.x - self.cx
        dy = node.y - self.cy
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Avoid divide-by-zero at the exact center
        if dist < 1e-12: 
            return 
            
        # Project onto the radius
        scale = self.r / dist
        node.x = self.cx + dx * scale
        node.y = self.cy + dy * scale


class Line(ParametricCurve):
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    def evaluate(self, t):
        x = self.x1 + (self.x2 - self.x1) * t
        y = self.y1 + (self.y2 - self.y1) * t
        return x, y

    def snap(self, node):
        # Project point (px, py) onto line segment from A to B
        px, py = node.x, node.y
        x1, y1 = self.x1, self.y1
        x2, y2 = self.x2, self.y2

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0: return

        # Calculate projection factor 'u'
        u = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)

        # Clamp 'u' to the segment [0, 1]
        u = max(0, min(1, u))

        # Closest point
        node.x = x1 + u * dx
        node.y = y1 + u * dy
'''


class Arc(ParametricCurve):
    def __init__(self, cx, cy, r, start_angle, end_angle):
        self.cx = cx
        self.cy = cy
        self.r = r
        # The caller provdes the angles in radians
        self.start = start_angle
        self.end = end_angle
        self.sweep = self.end - self.start

    def evaluate(self, t):
        # t goes from 0.0 to 1.0
        theta = self.start + (self.sweep * t)
        x = self.cx + self.r * math.cos(theta)
        y = self.cy + self.r * math.sin(theta)
        return x, y

    def snap(self, node):
        # 1. Calculate angle of the node relative to center
        dx = node.x - self.cx
        dy = node.y - self.cy
        target_angle = math.atan2(dy, dx) # Result is -pi to +pi
        
        # 2. Normalize angles to 0..2pi for easier comparison, or handle the wrap-around logic.
        #    (Simplified "snap to radius" for now, ignoring endpoints for robustness)
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0: return
        
        scale = self.r / dist
        node.x = self.cx + dx * scale
        node.y = self.cy + dy * scale
        
        
class CompositeCurve(ParametricCurve):
    def __init__(self, segments):
        """
        segments: A list of ParametricCurve objects (Lines, Arcs, etc.)
                  connected end-to-end.
        """
        self.segments = segments

    def evaluate(self, t):
        # Map t (0.0 to 1.0) to the specific segment.
        # Example: If we have 4 segments, t=0.25 is the end of segment 1.
        
        num_segs = len(self.segments)
        if t >= 1.0: 
            t = 0.999999 # Clamp to avoid index error
            
        # Find which segment 't' falls into
        seg_idx = int(t * num_segs)
        
        # Calculate local 't' for that specific segment (0.0 to 1.0)
        # Global t=0.6 in a 2-segment curve -> Local t=0.2 in segment 2
        seg_t = (t * num_segs) - seg_idx
        
        return self.segments[seg_idx].evaluate(seg_t)

    def snap(self, node):
        # To snap, we check distance to ALL segments and pick the closest one.
        best_dist = float('inf')
        best_x, best_y = node.x, node.y
        
        # This is a temporary dummy node to test snapping on sub-curves
        test_node = type(node)(0, node.x, node.y) 

        for seg in self.segments:
            # Reset test node position
            test_node.x, test_node.y = node.x, node.y
            
            # Snap to this segment
            seg.snap(test_node)
            
            # Check distance
            dist = (test_node.x - node.x)**2 + (test_node.y - node.y)**2
            
            if dist < best_dist:
                best_dist = dist
                best_x, best_y = test_node.x, test_node.y
        
        # Apply the winner
        node.x = best_x
        node.y = best_y
        

class CubicCurve(ParametricCurve):
    def __init__(self, x_start, x_end, a, b, c, d):
        """
        Defines a curve y = ax^3 + bx^2 + cx + d
        between x_start and x_end.
        """
        self.x0 = x_start
        self.x1 = x_end
        self.a, self.b, self.c, self.d = a, b, c, d

    def evaluate(self, t):
        # Map t (0..1) to x (x0..x1)
        x = self.x0 + (self.x1 - self.x0) * t
        
        # Calculate y based on the equation
        y = (self.a * x**3) + (self.b * x**2) + (self.c * x) + self.d
        return x, y

    def snap(self, node):
        # For complex equations, an analytical projection is hard.
        # We use a numerical search (Newton's method or simple scan).
        # A simple scan is robust enough for now:
        best_t = 0
        min_dist = float('inf')
        
        # Check 20 points along the curve to find the rough neighborhood
        for i in range(21):
            t = i / 20.0
            x, y = self.evaluate(t)
            dist = (node.x - x)**2 + (node.y - y)**2
            if dist < min_dist:
                min_dist = dist
                best_t = t
                
        # Refine (simple optimization step)
        # (For a production code, you'd use scipy.optimize, but this is fine)
        x, y = self.evaluate(best_t)
        node.x = x
        node.y = y        
        
 

class Polygon(CompositeCurve):
    def __init__(self, vertices):
        """
        vertices: A list of (x, y) tuples. 
                  e.g. [(0,0), (10,0), (5,5)]
        """
        if len(vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")

        segments = []
        
        # Loop through vertices and connect i to i+1
        for i in range(len(vertices)):
            curr_p = vertices[i]
            # Wrap around to the start for the last segment
            next_p = vertices[(i + 1) % len(vertices)]
            
            # Create the Line segment
            line = Line(curr_p[0], curr_p[1], next_p[0], next_p[1])
            segments.append(line)
            
        # Initialize the parent CompositeCurve with these lines
        super().__init__(segments)
        


class RegularPolygon(Polygon):
    def __init__(self, center_x, center_y, radius, n_sides):
        """
        Creates a regular N-sided polygon (Hexagon, Octagon, etc.)
        """
        vertices = []
        angle_step = 2 * math.pi / n_sides
        
        for i in range(n_sides):
            theta = i * angle_step
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            vertices.append((x, y))
            
        # Pass the calculated vertices to the parent Polygon constructor
        super().__init__(vertices)


        
        
        