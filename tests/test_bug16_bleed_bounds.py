"""
Bug 16 Verification Test: No Bounds Check in Bleed Layer Effect

Tests that bleed_layer_effect can produce values > 1.0 when:
- strength is high
- mask has high values
- blurred values are large

Expected: Bug reproduces (values > 1.0)
After fix: Values clamped to [0, 1]
"""

import torch
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from autoforge.Helper.OptimizerHelper import bleed_layer_effect


def test_bug16_bleed_exceeds_bounds():
    """
    Test case 1: High strength with high mask values
    Should produce values > 1.0 (bug) or be clamped to 1.0 (fixed)
    """
    print("=" * 70)
    print("Bug 16 Verification: Bleed Layer Effect Bounds Check")
    print("=" * 70)

    # Create a mask with high values (all 0.9)
    mask = torch.ones(5, 5) * 0.9

    # Apply high strength bleed effect
    strength = 0.5  # High strength

    result = bleed_layer_effect(mask, strength=strength)

    max_val = result.max().item()
    min_val = result.min().item()

    print(f"\nTest Case 1: Uniform high mask (0.9) with strength={strength}")
    print(f"  Input mask range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    print(f"  Output range: [{min_val:.4f}, {max_val:.4f}]")

    # Result might be [1, H, W] if input was 2D, so handle both cases
    if result.dim() == 3:
        center_val = result[0, 2, 2].item()
    else:
        center_val = result[2, 2].item()
    print(f"  Center pixel value: {center_val:.4f}")

    # Expected without clamp: 0.9 + 0.5 * (8 * 0.9 / 8) = 0.9 + 0.5 * 0.9 = 1.35
    expected_unclamped = 0.9 + 0.5 * 0.9
    print(f"  Expected (unclamped): {expected_unclamped:.4f}")

    if max_val > 1.0:
        print(f"\n  ❌ BUG CONFIRMED: Maximum value {max_val:.4f} exceeds 1.0!")
        print(f"     This violates opacity bounds and causes rendering artifacts.")
        return False
    else:
        print(f"\n  ✅ FIXED: Maximum value {max_val:.4f} is clamped to [0,1]")
        return True


def test_bug16_edge_case_high_strength():
    """
    Test case 2: Very high strength that would definitely exceed bounds
    """
    print("\n" + "-" * 70)
    print("Test Case 2: Very high strength")
    print("-" * 70)

    # Create a checkerboard pattern
    mask = torch.zeros(7, 7)
    mask[::2, ::2] = 1.0  # Alternating 1.0 and 0.0

    # Very high strength
    strength = 2.0

    result = bleed_layer_effect(mask, strength=strength)

    max_val = result.max().item()
    min_val = result.min().item()

    print(f"  Input: Checkerboard pattern [0, 1]")
    print(f"  Strength: {strength}")
    print(f"  Output range: [{min_val:.4f}, {max_val:.4f}]")

    if max_val > 1.0:
        print(f"\n  ❌ BUG CONFIRMED: Maximum value {max_val:.4f} exceeds 1.0!")
        return False
    elif min_val < 0.0:
        print(f"\n  ❌ BUG CONFIRMED: Minimum value {min_val:.4f} below 0.0!")
        return False
    else:
        print(f"\n  ✅ FIXED: Values properly clamped to [0,1]")
        return True


def test_bug16_3d_tensor():
    """
    Test case 3: 3D tensor (multi-layer) with high values
    """
    print("\n" + "-" * 70)
    print("Test Case 3: Multi-layer 3D tensor")
    print("-" * 70)

    # Create a 3D mask [L, H, W] with high values
    L, H, W = 3, 4, 4
    mask = torch.ones(L, H, W) * 0.95

    strength = 0.4

    result = bleed_layer_effect(mask, strength=strength)

    max_val = result.max().item()
    min_val = result.min().item()

    print(f"  Input shape: {mask.shape}, all values = 0.95")
    print(f"  Strength: {strength}")
    print(f"  Output range: [{min_val:.4f}, {max_val:.4f}]")

    # Expected without clamp: 0.95 + 0.4 * 0.95 = 1.33
    expected_unclamped = 0.95 + 0.4 * 0.95
    print(f"  Expected (unclamped): {expected_unclamped:.4f}")

    if max_val > 1.0:
        print(f"\n  ❌ BUG CONFIRMED: Maximum value {max_val:.4f} exceeds 1.0!")
        return False
    else:
        print(f"\n  ✅ FIXED: Maximum value {max_val:.4f} is clamped to [0,1]")
        return True


def test_bug16_realistic_scenario():
    """
    Test case 4: Realistic scenario with opacity masks from optimization
    """
    print("\n" + "-" * 70)
    print("Test Case 4: Realistic optimization scenario")
    print("-" * 70)

    # Simulate what might come from sigmoid activation (close to 1.0)
    # In actual optimization, sigmoid can produce values very close to 1.0
    mask = torch.sigmoid(torch.randn(10, 10) + 3.0)  # Bias toward high values

    strength = 0.1  # Default strength used in code

    result = bleed_layer_effect(mask, strength=strength)

    max_val = result.max().item()
    min_val = result.min().item()
    avg_val = result.mean().item()

    print(f"  Input: Sigmoid-activated random mask")
    print(f"  Input range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    print(f"  Strength: {strength}")
    print(f"  Output range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Output mean: {avg_val:.4f}")

    violations = (result > 1.0).sum().item() + (result < 0.0).sum().item()
    total = result.numel()

    print(f"  Pixels violating [0,1]: {violations}/{total}")

    if violations > 0:
        print(f"\n  ❌ BUG CONFIRMED: {violations} pixels violate bounds!")
        print(f"     In rendering, this causes invalid opacity values.")
        return False
    else:
        print(f"\n  ✅ FIXED: All pixels within [0,1] bounds")
        return True


if __name__ == "__main__":
    print("\nRunning Bug 16 verification tests...\n")

    test1_passed = test_bug16_bleed_exceeds_bounds()
    test2_passed = test_bug16_edge_case_high_strength()
    test3_passed = test_bug16_3d_tensor()
    test4_passed = test_bug16_realistic_scenario()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(
        f"Test 1 (Uniform high mask): {'✅ PASSED' if test1_passed else '❌ FAILED (bug exists)'}"
    )
    print(
        f"Test 2 (High strength): {'✅ PASSED' if test2_passed else '❌ FAILED (bug exists)'}"
    )
    print(
        f"Test 3 (3D tensor): {'✅ PASSED' if test3_passed else '❌ FAILED (bug exists)'}"
    )
    print(
        f"Test 4 (Realistic scenario): {'✅ PASSED' if test4_passed else '❌ FAILED (bug exists)'}"
    )

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed

    if not all_passed:
        print("\n" + "🔴" * 35)
        print("BUG 16 CONFIRMED: bleed_layer_effect needs bounds clamping")
        print("Fix: Add torch.clamp(result, 0.0, 1.0) before returning")
        print("🔴" * 35)
        sys.exit(1)
    else:
        print("\n" + "🟢" * 35)
        print("BUG 16 FIXED: All bounds checks passing")
        print("🟢" * 35)
        sys.exit(0)
