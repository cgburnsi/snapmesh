"""
snapfvm/display.py
------------------
Standardized console output for SnapFVM simulations.
Provides consistent headers, section breaks, and tabular data logging.
"""
import sys
import time
import numpy as np

class SimulationDisplay:
    def __init__(self, title, context_info):
        """
        Initialize the display manager.
        
        Args:
            title (str): Name of the script or simulation (e.g. "Rocket Nozzle")
            context_info (str): Physics config (e.g. "Euler 2D | Order=2")
        """
        self.title = title
        self.context = context_info
        self.start_time = time.time()
        self._col_widths = []
        self._headers = []

    def header(self):
        """Prints the Minimalist Header."""
        width = 70
        print("-" * width)
        print(f"SnapFVM :: {self.title}")
        print(f"Config  :: {self.context}")
        print("-" * width + "\n")

    def section(self, name):
        """Prints a visual break for a new phase (e.g., 'Mesh Generation')."""
        print(f"--- {name} ---")

    def setup_stats_columns(self, headers, widths=None):
        """
        Defines the columns for the iteration log.
        
        Args:
            headers (list of str): Column names, e.g. ["Iter", "Time", "Resid"]
            widths (list of int, optional): Width of each column. Defaults to 12.
        """
        self._headers = headers
        if widths is None:
            self._col_widths = [12] * len(headers)
        else:
            self._col_widths = widths
            
        # Print the table header row
        # We add a blank line before the table for clarity
        print("") 
        header_str = "  ".join([h.rjust(w) for h, w in zip(self._headers, self._col_widths)])
        print(header_str)
        print("-" * len(header_str))

    def log_stats(self, *args):
        """
        Logs a row of data matching the columns defined in setup_stats_columns.
        Automatically formats floats (scientific vs fixed) based on magnitude.
        """
        if len(args) != len(self._col_widths):
            # Fallback if user passes wrong number of args
            print(f" [Display Error] Expected {len(self._col_widths)} args, got {len(args)}: {args}")
            return

        row_str = []
        for val, width in zip(args, self._col_widths):
            if isinstance(val, int):
                s = f"{val:d}".rjust(width)
            elif isinstance(val, (float, np.floating)): # Handle numpy floats too
                # Intelligent formatting
                abs_val = abs(val)
                if abs_val == 0:
                     s = f"{0.0:.4f}".rjust(width)
                elif abs_val < 1e-2 or abs_val >= 1e5:
                    s = f"{val:.2e}".rjust(width)
                else:
                    s = f"{val:.4f}".rjust(width)
            else:
                s = str(val).rjust(width)
            row_str.append(s)
            
        print("  ".join(row_str))

    def success(self, message="Simulation Complete"):
        """Prints the success footer with elapsed time."""
        elapsed = time.time() - self.start_time
        print(f"\n>> {message} ({elapsed:.2f}s)\n")

    def error(self, message):
        """Prints a critical error message."""
        print(f"\n!! CRITICAL ERROR: {message} !!\n")

# -- Convenience Alias --
# So you can just import 'Display' if you prefer brevity
Display = SimulationDisplay