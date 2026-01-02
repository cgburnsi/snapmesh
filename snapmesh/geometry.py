"""
snapmesh/geometry.py
--------------------
Defines geometric constraints.
Nodes attached to these objects can 'snap' to them to maintain shape fidelity.
"""
import numpy as np
from abc import ABC, abstractmethod

# Topological Tags
TAG_SMOOTH = 0
TAG_CORNER = 1

class GeometricConstraint(ABC):
    """ Abstract Base Class for all geometric constraints. """
    
    @abstractmethod
    def snap(self, node):
        """ Modifies node.x and node.y in-place to lie on the curve. """
        pass
    
    @abstractmethod
    def evaluate(self, t):
        """ Returns (x, y) at parameter t [0,1]. """
        pass
        
    @abstractmethod
    def length(self):
        """ Returns total arc length. """
        pass

    def discretize(self, sizing_func, min_points=2):
        """
        Generates points along the curve based on local sizing.
        Returns:
            points: List of (x, y) tuples
            tags:   List of flags (CORNER or SMOOTH)
        """
        points = []
        
        # Start Point
        points.append(self.evaluate(0.0))
        
        L = self.length()
        if L < 1e-12:
            return np.array(points), np.array([TAG_CORNER])

        max_dt = 1.0 / max(1, (min_points - 1))
        
        t = 0.0
        max_iter = 10000
        it = 0
        
        while t < 1.0:
            p_curr = self.evaluate(t)
            h_req = sizing_func(p_curr[0], p_curr[1])
            
            # 0.9 safety factor
            dt_req = (0.9 * h_req) / L
            dt = min(dt_req, max_dt)
            
            t += dt
            if t >= 1.0: break
                
            points.append(self.evaluate(t))
            it += 1
            if it > max_iter: break
            
        # End Point
        points.append(self.evaluate(1.0))
        
        pts_array = np.array(points)
        
        # Default Tagging: Endpoints are Corners
        tags = np.full(len(pts_array), TAG_SMOOTH, dtype=int)
        tags[0] = TAG_CORNER
        tags[-1] = TAG_CORNER
        
        return pts_array, tags

class LineSegment(GeometricConstraint):
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.vec = self.p2 - self.p1
        self._len = np.linalg.norm(self.vec)
        self.len_sq = self._len ** 2

    def length(self):
        return self._len

    def evaluate(self, t):
        return self.p1 + t * self.vec

    def snap(self, node):
        p = node.to_array()
        ap = p - self.p1
        if self.len_sq > 0:
            t = np.clip(np.dot(ap, self.vec) / self.len_sq, 0.0, 1.0)
            closest = self.p1 + t * self.vec
        else:
            closest = self.p1
        node.update_from_array(closest)

class Circle(GeometricConstraint):
    """ A full 360-degree circle. """
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.r = float(radius)
        self._len = 2 * np.pi * self.r

    def length(self):
        return self._len

    def evaluate(self, t):
        # Map t=0..1 to 0..2pi
        theta = t * 2.0 * np.pi
        x = self.center[0] + self.r * np.cos(theta)
        y = self.center[1] + self.r * np.sin(theta)
        return np.array([x, y])

    def snap(self, node):
        p = node.to_array()
        vec = p - self.center
        dist = np.linalg.norm(vec)
        if dist < 1e-12:
            # Degenerate: snap to arbitrary point
            closest = self.center + np.array([self.r, 0.0])
        else:
            closest = self.center + vec * (self.r / dist)
        node.update_from_array(closest)

class Arc(GeometricConstraint):
    def __init__(self, center, r, start_angle, end_angle):
        self.center = np.array(center, dtype=float)
        self.r = float(r)
        self.t1 = float(start_angle)
        self.t2 = float(end_angle)
        self.angle_diff = self.t2 - self.t1
        self._len = abs(self.angle_diff) * self.r
        
    def length(self):
        return self._len

    def evaluate(self, t):
        theta = self.t1 + t * self.angle_diff
        x = self.center[0] + self.r * np.cos(theta)
        y = self.center[1] + self.r * np.sin(theta)
        return np.array([x, y])

    def snap(self, node):
        p = node.to_array()
        vec = p - self.center
        dist = np.linalg.norm(vec)
        if dist < 1e-12:
            closest = self.evaluate(0.0)
        else:
            closest = self.center + vec * (self.r / dist)
        node.update_from_array(closest)

class PolyLine(GeometricConstraint):
    """ A chain of segments treated as one constraint. """
    def __init__(self, segments):
        self.segments = segments
        self.lengths = [s.length() for s in segments]
        self.total_len = sum(self.lengths)
        
    def length(self):
        return self.total_len
        
    def evaluate(self, t):
        if t <= 0.0: return self.segments[0].evaluate(0.0)
        if t >= 1.0: return self.segments[-1].evaluate(1.0)
        
        target_len = t * self.total_len
        current_len = 0.0
        for i, seg_len in enumerate(self.lengths):
            if current_len + seg_len >= target_len:
                local_t = (target_len - current_len) / seg_len if seg_len > 0 else 0.0
                return self.segments[i].evaluate(local_t)
            current_len += seg_len
        return self.segments[-1].evaluate(1.0)

    def snap(self, node):
        best_dist_sq = float('inf')
        best_pos = None
        orig = node.to_array()
        
        for seg in self.segments:
            seg.snap(node)
            curr = node.to_array()
            d2 = np.sum((curr - orig)**2)
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_pos = curr
            node.update_from_array(orig)
            
        if best_pos is not None:
            node.update_from_array(best_pos)
            
    def discretize(self, sizing_func, min_points=2):
        # PolyLine handles sub-segment corners
        all_pts = []
        all_tags = []
        
        for i, seg in enumerate(self.segments):
            pts, tags = seg.discretize(sizing_func, min_points)
            if i > 0:
                pts = pts[1:]
                tags = tags[1:]
            
            tags[0] = TAG_CORNER
            tags[-1] = TAG_CORNER
            
            all_pts.append(pts)
            all_tags.append(tags)
            
        if not all_pts: return np.empty((0,2)), np.empty(0, int)
        return np.vstack(all_pts), np.concatenate(all_tags)