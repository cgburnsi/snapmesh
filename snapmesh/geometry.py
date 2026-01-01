"""
snapmesh/geometry.py
--------------------
Object-Oriented Geometric Primitives.
Supports:
1. Parametric Evaluation (evaluate(t)) -> For Meshing
2. Projection (snap(node))             -> For Constraints
"""
import numpy as np
from abc import ABC, abstractmethod
import math

class ParametricCurve(ABC):
    """ Abstract base for all 1D Curves. """
    
    @abstractmethod
    def evaluate(self, t):
        """ Returns np.array([x, y]) at parameter t (0.0 <= t <= 1.0). """
        pass

    @abstractmethod
    def snap(self, node):
        """ Modifies the node's (x, y) position in-place. """
        pass
        
    @abstractmethod
    def length(self):
        """ Returns the total arc length of the curve. """
        pass

    def discretize(self, target_h):
        """ 
        Uniform discretization: Returns points spaced by roughly 'target_h'. 
        """
        L = self.length()
        # Ensure at least 2 points (Start/End)
        n = max(2, int(np.round(L / target_h)))
        t_vals = np.linspace(0.0, 1.0, n)
        return np.array([self.evaluate(t) for t in t_vals])

    def discretize_adaptive(self, sizing_func):
        """
        Adaptive discretization: Steps along the curve based on local h requirements.
        sizing_func: f(x, y) -> target_h
        """
        points = []
        t = 0.0
        
        # Add Start Point
        points.append(self.evaluate(t))
        
        L = self.length()
        if L < 1e-12:
            return np.array(points)

        # Safety: Limit iterations
        max_iter = 10000
        it = 0
        
        while t < 1.0:
            p_curr = self.evaluate(t)
            h_req = sizing_func(p_curr[0], p_curr[1])
            
            # Step size in parameter space (dt = dx / L)
            # Conservative step (0.9) to catch curvature
            dt = (0.9 * h_req) / L
            
            t += dt
            if t >= 1.0: break
                
            points.append(self.evaluate(t))
            it += 1
            if it > max_iter: break
            
        # Force End Point
        points.append(self.evaluate(1.0))
        return np.array(points)


class LineSegment(ParametricCurve):
    """ 
    A straight line connecting two points (p1, p2).
    """
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        
        self.vec = self.p2 - self.p1
        self._len = np.linalg.norm(self.vec)
        self.len_sq = self._len ** 2
        
        if self.len_sq == 0:
            # Degenerate line handling could go here, but raising error is safer for now
            pass 

    def length(self):
        return self._len

    def evaluate(self, t):
        """ Returns point at t (0.0 = p1, 1.0 = p2). """
        return self.p1 + t * self.vec

    def snap(self, node):
        """ Projects node onto the SEGMENT. Clamps to endpoints. """
        p = node.to_array()
        ap = p - self.p1
        
        if self.len_sq > 0:
            t = np.dot(ap, self.vec) / self.len_sq
            t = np.clip(t, 0.0, 1.0)
            closest = self.p1 + t * self.vec
        else:
            closest = self.p1
            
        node.update_from_array(closest)

    def __repr__(self):
        return f"LineSegment(p1={self.p1}, p2={self.p2})"


class Circle(ParametricCurve):
    """ A full circle defined by center and radius. """
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=np.float64)
        self.r = float(radius)
        self._len = 2 * np.pi * self.r

    def length(self):
        return self._len

    def evaluate(self, t):
        # Map t to angle (0..2pi)
        theta = t * 2.0 * np.pi
        x = self.center[0] + self.r * np.cos(theta)
        y = self.center[1] + self.r * np.sin(theta)
        return np.array([x, y])

    def snap(self, node):
        p = node.to_array()
        vec = p - self.center
        dist = np.linalg.norm(vec)
        
        if dist < 1e-12:
            closest = self.center + np.array([self.r, 0])
        else:
            closest = self.center + vec * (self.r / dist)
        
        node.update_from_array(closest)


class Arc(ParametricCurve):
    """
    A circular arc defined by center, radius, and start/end angles (Radians).
    """
    def __init__(self, center, r, start_angle, end_angle):
        self.center = np.array(center, dtype=np.float64)
        self.r = float(r)
        self.t1 = float(start_angle)
        self.t2 = float(end_angle)
        self.angle_diff = self.t2 - self.t1
        self._len = abs(self.angle_diff) * self.r
        
    def length(self):
        return self._len

    def evaluate(self, t):
        # Map t=[0,1] to angle=[t1, t2]
        theta = self.t1 + t * self.angle_diff
        x = self.center[0] + self.r * np.cos(theta)
        y = self.center[1] + self.r * np.sin(theta)
        return np.array([x, y])

    def snap(self, node):
        # 1. Project to infinite circle first
        p = node.to_array()
        vec = p - self.center
        dist = np.linalg.norm(vec)
        
        if dist < 1e-12:
            # Degenerate node at center: snap to start of arc
            closest = self.evaluate(0.0)
        else:
            # Project to circle
            p_circ = self.center + vec * (self.r / dist)
            
            # 2. Check if p_circ is within the arc angles
            # (Simple approach: calculate t for p_circ)
            theta_p = np.arctan2(p_circ[1]-self.center[1], p_circ[0]-self.center[0])
            
            # Unwrap theta_p to be close to t1
            # This logic can be complex for generic wrapping. 
            # For now, we assume simple projection is 'good enough' for basic snapping
            # or rely on the user to place nodes reasonably close.
            closest = p_circ

        node.update_from_array(closest)


class PolyLine(ParametricCurve):
    """ 
    A chain of curves (Segments, Arcs) treated as one continuous object. 
    """
    def __init__(self, segments):
        self.segments = segments
        self.lengths = [s.length() for s in segments]
        self.total_len = sum(self.lengths)
        
    def length(self):
        return self.total_len
        
    def evaluate(self, t):
        # Map t=[0,1] to the correct segment
        if t <= 0.0: return self.segments[0].evaluate(0.0)
        if t >= 1.0: return self.segments[-1].evaluate(1.0)
        
        target_len = t * self.total_len
        current_len = 0.0
        
        for i, seg_len in enumerate(self.lengths):
            if current_len + seg_len >= target_len:
                local_dist = target_len - current_len
                local_t = local_dist / seg_len if seg_len > 0 else 0.0
                return self.segments[i].evaluate(local_t)
            current_len += seg_len
            
        return self.segments[-1].evaluate(1.0)

    def snap(self, node):
        # Snap to whichever segment is closest
        best_dist_sq = float('inf')
        best_pos = None
        
        # We need a temp node to probe the segments without modifying the real node yet
        # or we just save/restore the state.
        original_pos = node.to_array()
        
        for seg in self.segments:
            # Snap the node speculatively
            seg.snap(node)
            curr_pos = node.to_array()
            
            dist_sq = np.sum((curr_pos - original_pos)**2)
            
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_pos = curr_pos
            
            # Reset for next iteration
            node.update_from_array(original_pos)
            
        # Apply winner
        if best_pos is not None:
            node.update_from_array(best_pos)
    
    def discretize_adaptive(self, sizing_func):
        all_pts = []
        for i, seg in enumerate(self.segments):
            pts = seg.discretize_adaptive(sizing_func)
            if i > 0:
                pts = pts[1:] # Avoid duplicate at join
            all_pts.append(pts)
            
        if not all_pts: return np.empty((0,2))
        return np.vstack(all_pts)