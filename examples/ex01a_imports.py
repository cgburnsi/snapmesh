"""
ex01a_imports.py
----------------
System Health Check.
Verifies that the main Snap Suite packages and critical third-party 
libraries are installed and accessible.
"""
import sys
import importlib

# Try to import the display tool first.
try:
    from snapcore.display import SimulationDisplay
    has_display = True
except ImportError:
    has_display = False
    print("!! CRITICAL ERROR: Could not import 'snapcore.display'. !!")

def run_check():
    if not has_display:
        return

    # 1. Initialize Display
    disp = SimulationDisplay("Snap Suite Examples", "EXAMPLE 01a - Environment Verification")
    disp.header()

    # 2. Define Modules to Verify (Top-Level Only)
    modules_to_check = [
        # --- Snap Suite ---
        ("snapcore",          "Core Utilities"),
        ("snapmesh",          "Geometry & Meshing"),
        ("snapfvm",           "Finite Volume Solver"),
        ("snapchem",          "Chemistry (Placeholder)"),
        
        # --- Critical Third Party ---
        ("numpy",             "Core Numerics"),
        ("matplotlib",        "Visualization"),
        ("scipy",             "Optimization/Roots"),
        ("numba",             "JIT Compiler") 
    ]

    disp.section("Module Verification")
    
    # We'll use the stats table to show Pass/Fail status nicely
    disp.setup_stats_columns(["Module", "Status", "Version/Path"], [25, 10, 15])

    all_passed = True
    
    for mod_name, desc in modules_to_check:
        status = "FAIL"
        path_info = "Not Found"
        
        try:
            # Dynamically import the module
            mod = importlib.import_module(mod_name)
            status = "OK"
            
            # 1. Try to get Version
            if hasattr(mod, '__version__'):
                path_info = f"v{mod.__version__}"
            
            # 2. If no version, fallback to File Path
            elif hasattr(mod, '__file__'):
                full_path = mod.__file__
                if len(full_path) > 35:
                    path_info = "..." + full_path[-32:]
                else:
                    path_info = full_path
            else:
                path_info = "(Built-in)"
                
        except ImportError:
            all_passed = False
            path_info = "Module Missing"
        except Exception as e:
            all_passed = False
            status = "ERROR"
            path_info = str(e)

        # Log the row
        disp.log_stats(mod_name, status, path_info)

    # 3. Final Summary
    print("\n")
    if all_passed:
        disp.success("System Ready")
    else:
        disp.error("System Verification Failed")

if __name__ == "__main__":
    run_check()