from autoforge.Modules.Optimizer import FilamentOptimizer
import inspect

# Get the source code of __init__
source = inspect.getsource(FilamentOptimizer.__init__)

# Check for validation
if "init_tau < self.final_tau" in source:
    print("✓ Validation code IS in the loaded module")
else:
    print("✗ Validation code NOT in the loaded module")

# Print key lines
print("\nSearching for tau-related lines:")
lines = source.split("\n")
for i, line in enumerate(lines):
    if "init_tau" in line or "final_tau" in line:
        print(f"  Line {i}: {line[:80]}")
