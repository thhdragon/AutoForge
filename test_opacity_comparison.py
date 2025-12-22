import torch
import numpy as np

# Old formula (Bug #15 original)
o, A, k, b = -1.2416557e-02, 9.6407950e-01, 3.4103447e01, -4.1554203e00

# Test with various thickness ratios
thick_ratios = torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

# Old formula
opac_old = o + (A * torch.log1p(k * thick_ratios) + b * thick_ratios)
print("Old formula opacities:")
print(f"Thick ratio | Opacity")
for tr, op in zip(thick_ratios.numpy(), opac_old.numpy()):
    print(f"{tr:8.2f}    | {op:8.4f}")

print("\n" + "=" * 40 + "\n")

# New formula (Beer-Lambert with k=3.0)
k_opacity = 3.0
opac_new = 1.0 - torch.exp(-k_opacity * thick_ratios)
print("New formula (k=3.0) opacities:")
print(f"Thick ratio | Opacity")
for tr, op in zip(thick_ratios.numpy(), opac_new.numpy()):
    print(f"{tr:8.2f}    | {op:8.4f}")

print("\n" + "=" * 40 + "\n")
print("Difference (new - old):")
diff = opac_new - opac_old
for tr, d in zip(thick_ratios.numpy(), diff.numpy()):
    print(f"{tr:8.2f}    | {d:+8.4f}")

print("\n" + "=" * 40 + "\n")
print("Old formula CLAMPED to [0,1]:")
opac_old_clamped = torch.clamp(opac_old, 0.0, 1.0)
for tr, op in zip(thick_ratios.numpy(), opac_old_clamped.numpy()):
    print(f"{tr:8.2f}    | {op:8.4f}")

print("\n" + "=" * 40 + "\n")
print("Comparison at key thick_ratios:")
print(
    f"At thick_ratio = 0.1: Old(clamped)={opac_old_clamped[1].item():.4f}, New={opac_new[1].item():.4f}"
)
print(
    f"At thick_ratio = 0.2: Old(clamped)={opac_old_clamped[2].item():.4f}, New={opac_new[2].item():.4f}"
)
print(
    f"At thick_ratio = 1.0 (TD): Old(clamped)={opac_old_clamped[4].item():.4f}, New={opac_new[4].item():.4f}"
)
print(f"\nHueForge expects ~95% opacity at TD")
print(f"New formula gives: {opac_new[4].item() * 100:.1f}%")
