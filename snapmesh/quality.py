"""
snapmesh/quality.py
-------------------
Tools for inspecting mesh geometric fidelity.
Calculates metrics like Aspect Ratio, Minimum Angle, and Area.
"""
import numpy as np
import matplotlib.pyplot as plt

class MeshQuality:
    """
    Inspector class for a Mesh object.
    
    Usage:
        inspector = MeshQuality(mesh)
        inspector.analyze()
        inspector.print_report()
        inspector.plot_histograms()
    """
    def __init__(self, mesh):
        self.mesh = mesh
        # Metric Storage
        self.areas = []
        self.min_angles = []
        self.aspect_ratios = []
        self.ids = [] 
        
        self._analyzed = False

    def analyze(self):
        """
        Iterates through all cells and computes metrics.
        """
        self.areas = []
        self.min_angles = []
        self.aspect_ratios = []
        self.ids = []
        
        for cell_id, cell in self.mesh.cells.items():
            area, min_ang, ar = self._compute_single_cell(cell)
            
            self.ids.append(cell_id)
            self.areas.append(area)
            self.min_angles.append(min_ang)
            self.aspect_ratios.append(ar)
            
        # Convert to numpy for easy stats
        self.areas = np.array(self.areas)
        self.min_angles = np.array(self.min_angles)
        self.aspect_ratios = np.array(self.aspect_ratios)
        self.ids = np.array(self.ids)
        
        self._analyzed = True

    def _compute_single_cell(self, cell):
        """ Helper: Returns (area, min_angle_deg, aspect_ratio) for one cell. """
        p1 = cell.n1.to_array()
        p2 = cell.n2.to_array()
        p3 = cell.n3.to_array()
        
        # Edge vectors
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p1 - p3
        
        # Lengths
        a = np.linalg.norm(v1)
        b = np.linalg.norm(v2)
        c = np.linalg.norm(v3)
        
        # Area (Heron's Formula)
        s = (a + b + c) / 2.0
        arg = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(max(0.0, arg))
        
        # Aspect Ratio
        if area > 1e-15:
            r_in = area / s
            R_circ = (a * b * c) / (4 * area)
            ar = R_circ / (2 * r_in)
        else:
            ar = 999.0 # Degenerate
            
        # Angles (Cosine Rule)
        angles = []
        for edge_len, adj1, adj2 in [(a, b, c), (b, a, c), (c, a, b)]:
            denom = 2 * adj1 * adj2
            if denom > 1e-15:
                cos_theta = (adj1**2 + adj2**2 - edge_len**2) / denom
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_theta)))
            else:
                angles.append(0.0)
                
        min_ang = min(angles) if angles else 0.0
        
        return area, min_ang, ar

    def print_report(self):
        """ Prints a summary to stdout. """
        if not self._analyzed: self.analyze()
        
        print(f"--- Mesh Quality Report ({len(self.ids)} Cells) ---")
        
        print(f"Area:")
        print(f"  Min: {self.areas.min():.2e} m^2")
        print(f"  Max: {self.areas.max():.2e} m^2")
        
        min_ang = self.min_angles.min()
        print(f"Min Angle: {min_ang:.2f} deg  ", end="")
        if min_ang < 10.0: print("[!] WARNING: Slivers Detected")
        elif min_ang < 20.0: print("[~] CAUTION: Low Quality")
        else: print("[OK] Good")
            
        max_ar = self.aspect_ratios.max()
        print(f"Max Aspect Ratio: {max_ar:.2f}  ", end="")
        if max_ar > 10.0: print("[!] WARNING: Highly Stretched")
        elif max_ar > 3.0: print("[~] CAUTION")
        else: print("[OK]")

    def plot_histograms(self):
        """ Visualizes the distribution of quality metrics. """
        if not self._analyzed: self.analyze()
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        # --- ROBUST HISTOGRAM HELPER ---
        def safe_hist(axis, data, color, title, xlabel, limit_line=None):
            if len(data) == 0: return
            
            # If all values are identical (Variance = 0), fixed bins crash.
            # We detect this and use a manual range.
            dmin, dmax = data.min(), data.max()
            if np.isclose(dmin, dmax):
                # Center the single value in a small window
                padding = max(1e-6, abs(dmin)*0.1) 
                bins = np.linspace(dmin - padding, dmax + padding, 10)
                axis.hist(data, bins=bins, color=color, edgecolor='black')
            else:
                # Normal case
                axis.hist(data, bins=20, color=color, edgecolor='black')
            
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            if limit_line:
                axis.axvline(limit_line, color='red', linestyle='--', label='Limit')
                axis.legend()

        # 1. Min Angle
        safe_hist(ax[0], self.min_angles, 'skyblue', 
                 "Minimum Angle (Target > 20°)", "Degrees", limit_line=20)
        
        # 2. Aspect Ratio
        safe_hist(ax[1], self.aspect_ratios, 'lightgreen', 
                 "Aspect Ratio (Target < 3.0)", "Ratio", limit_line=3.0)
        
        # 3. Area
        safe_hist(ax[2], self.areas, 'salmon', 
                 "Cell Areas", "Area [m^2]")
        
        plt.tight_layout()
        plt.show()