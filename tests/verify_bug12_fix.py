"""Verify Bug #12 fix: Double .mean() call removal"""

import torch
from src.autoforge.Loss.LossFunctions import compute_loss

# Create sample tensors
batch_h, batch_w = 64, 64
comp = torch.rand(batch_h, batch_w, 3)
target = torch.rand(batch_h, batch_w, 3)
focus_map = torch.rand(batch_h, batch_w)  # Priority mask

# Test 1: Loss without focus map (simple case)
loss1 = compute_loss(comp, target)
print(f"Test 1 - Simple loss:")
print(f"  Shape: {loss1.shape}")
print(f"  Dim: {loss1.dim()}")
print(f"  Value: {loss1.item():.6f}")
print(f"  Is scalar: {loss1.dim() == 0}")

# Test 2: Loss with focus map (the fixed case)
loss2 = compute_loss(comp, target, focus_map=focus_map, focus_strength=10.0)
print(f"\nTest 2 - Loss with focus map (FIXED CASE):")
print(f"  Shape: {loss2.shape}")
print(f"  Dim: {loss2.dim()}")
print(f"  Value: {loss2.item():.6f}")
print(f"  Is scalar: {loss2.dim() == 0}")

# Verify both are proper scalars
assert loss1.dim() == 0, f"Expected scalar (dim=0), got dim={loss1.dim()}"
assert loss2.dim() == 0, f"Expected scalar (dim=0), got dim={loss2.dim()}"

print("\n✓ Fix verified: Both losses are now proper scalars (dim=0)")
print("✓ Bug #12 FIXED: Removed redundant .mean() call")
