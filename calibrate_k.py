import torch
import numpy as np

# Typical scenario: h=0.04mm, TD=3mm
h = 0.04
TD = 3.0
thick_ratio_single_layer = h / TD
print(f"Single layer thick_ratio: {thick_ratio_single_layer:.6f}")

# Old formula (clamped) at various thick_ratios
o, A, k_old, b = -1.2416557e-02, 9.6407950e-01, 3.4103447e01, -4.1554203e00

# Test thick_ratios from 0.01 to 0.2 (1-20 layers of 0.04mm with TD=3mm)
thick_ratios = torch.linspace(0.01, 0.2, 20)
opac_old = o + (A * torch.log1p(k_old * thick_ratios) + b * thick_ratios)
opac_old_clamped = torch.clamp(opac_old, 0.0, 1.0)

print("\nOld formula (clamped) for small thick_ratios:")
print("Layers | thick_ratio | Opacity")
for i, (tr, op) in enumerate(zip(thick_ratios.numpy(), opac_old_clamped.numpy())):
    layers = tr / thick_ratio_single_layer
    print(f"{layers:6.1f} | {tr:11.6f} | {op:7.4f}")

# Now let's find k that matches the old formula's behavior at key points
# At thick_ratio ~ 0.1 (about 7.5 layers), old gave ~1.0 opacity
# At thick_ratio ~ 0.5 (about 37.5 layers), old gave ~0.70 opacity

# Try different k values
print("\n" + "=" * 60)
print("Finding k that matches old formula behavior:\n")

target_ratios = torch.tensor([0.01, 0.02, 0.05, 0.1, 0.2])
print("Testing different k values:")
print(
    f"{'k':>6s} | thick_ratios: "
    + " ".join([f"{tr:.2f}" for tr in target_ratios.numpy()])
)
print("-" * 60)

for k_test in [3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]:
    opac_new = 1.0 - torch.exp(-k_test * target_ratios)
    opac_str = " ".join([f"{op:.2f}" for op in opac_new.numpy()])
    print(f"{k_test:6.1f} | {opac_str}")

# Get old formula values for comparison
opac_old_targets = o + (A * torch.log1p(k_old * target_ratios) + b * target_ratios)
opac_old_targets_clamped = torch.clamp(opac_old_targets, 0.0, 1.0)
opac_str = " ".join([f"{op:.2f}" for op in opac_old_targets_clamped.numpy()])
print(f"{'OLD':>6s} | {opac_str}")

print("\n" + "=" * 60)
print("\nRecommendation:")
print("The old formula gave ~100% opacity at thick_ratio=0.1")
print("For 1 - exp(-k*0.1) = 0.95, we need k ≈ 30")
print("For 1 - exp(-k*0.1) = 1.00, we need k >> 30")
