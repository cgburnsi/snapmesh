import os
import shutil
import re

def migrate_library_structure():
    print("--- Starting Library Refactor ---")
    
    # 1. Create New Directories
    dirs_to_make = [
        "snapcore", 
        "snapchem", 
        "snapfvm/grid"
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)
        # Add __init__.py if missing
        if not os.path.exists(f"{d}/__init__.py"):
            open(f"{d}/__init__.py", 'w').close()
            
    # 2. Move Files (The Big Shift)
    moves = [
        # Source -> Destination
        ("unit_convert.py", "snapcore/units.py"),
        ("snapfvm/display.py", "snapcore/display.py"),
        ("snapmesh/grid.py", "snapfvm/grid/unstruct.py"), 
    ]
    
    for src, dst in moves:
        if os.path.exists(src):
            print(f"Moving: {src} -> {dst}")
            shutil.move(src, dst)
        else:
            print(f"Skipping: {src} (Not found)")

    # 3. Create wrapper for Grid to maintain cleaner imports
    # We moved grid.py to snapfvm/grid/unstruct.py, but we want users 
    # to type 'from snapfvm.grid import Grid'
    with open("snapfvm/grid/__init__.py", "w") as f:
        f.write("from .unstruct import Grid\n")

    print("\n--- Updating Example Imports ---")
    
    # 4. Update Imports in all Python files
    # Regex Patterns for replacement
    replacements = [
        (r"import unit_convert", r"import snapcore.units"),
        (r"from snapfvm\.display", r"from snapcore.display"),
        (r"from snapmesh\.grid", r"from snapfvm.grid"),
        # Fix usages of the renamed modules
        (r"unit_convert\.", r"snapcore.units."),
        (r"cv\.convert", r"snapcore.units.convert"), 
    ]
    
    # Scan examples folder
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        for filename in os.listdir(examples_dir):
            if filename.endswith(".py"):
                filepath = os.path.join(examples_dir, filename)
                
                with open(filepath, 'r') as f:
                    content = f.read()
                
                new_content = content
                for pattern, repl in replacements:
                    new_content = re.sub(pattern, repl, new_content)
                
                if new_content != content:
                    print(f"Updating imports in: {filename}")
                    with open(filepath, 'w') as f:
                        f.write(new_content)
                        
    print("\n--- Refactor Complete ---")

if __name__ == "__main__":
    migrate_library_structure()