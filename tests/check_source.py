import types
import torch
import numpy as np
from autoforge.Modules.Optimizer import FilamentOptimizer
import inspect

# Get the source code of __init__
source = inspect.getsource(FilamentOptimizer.__init__)

# Check for validation
if "init_tau < self.final_tau" in source:
    print("✓ Validation code IS in the loaded module")
else:
    print("✗ Validation code NOT in the loaded module")

# Print the relevant section
lines = source.split("\n")
for i, line in enumerate(lines):
    if "init_tau" in line.lower() or "final_tau" in line.lower():
        start = max(0, i - 2)
        end = min(len(lines), i + 3)
        print(f"\nLines {start}-{end}:")
        for j in range(start, end):
            print(f"  {j}: {lines[j]}")
