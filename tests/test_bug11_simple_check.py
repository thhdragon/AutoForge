"""Simple check that Bug 11 fix works"""

import sys
import torch

sys.path.insert(0, "src")

from autoforge.Helper.PruningHelper import smooth_coplanar_faces

# Test 1: Sharp feature should cause smoothing
print("Test 1: Sharp feature")
height_logits = torch.zeros(5, 5)
height_logits[:, 2] = 10.0
print("Input:")
print(height_logits)

result = smooth_coplanar_faces(height_logits, angle_threshold=30)
print("\nSmoothed (angle_threshold=30):")
print(result)

result2 = smooth_coplanar_faces(height_logits, angle_threshold=60)
print("\nSmoothed (angle_threshold=60):")
print(result2)

# Check if smoothing occurred
if torch.allclose(result, height_logits):
    print("WARNING: No smoothing with threshold=30")
else:
    print("✓ Smoothing occurred with threshold=30")

if torch.allclose(result2, height_logits):
    print("WARNING: No smoothing with threshold=60")
else:
    print("✓ Smoothing occurred with threshold=60")

# Test 2: Planar surface
print("\n" + "=" * 60)
print("Test 2: Planar surface")
planar = torch.ones(5, 5) * 7.0
result_planar = smooth_coplanar_faces(planar, angle_threshold=5)
print(f"Input: all 7.0")
print(f"Output: {result_planar}")
print(f"Std: {torch.std(result_planar):.6f}")

# Test 3: Gradient
print("\n" + "=" * 60)
print("Test 3: Gradient surface")
gradient = torch.arange(25, dtype=torch.float32).reshape(5, 5)
print("Input:")
print(gradient)
result_grad = smooth_coplanar_faces(gradient, angle_threshold=45)
print("\nSmoothed:")
print(result_grad)
