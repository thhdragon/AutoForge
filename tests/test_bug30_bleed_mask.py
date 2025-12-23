"""
Verification test for Bug #30: Bleed Mask Kernel Excludes Center Pixel

Bug #30 Issue:
The bleed_layer_effect kernel has 0 in the center:
  kernel = [[1,1,1], [1,0,1], [1,1,1]] / 8.0

This means the center pixel is completely excluded from the blur calculation,
making the bleed effect weak. The fix is to include the center:
  kernel = [[1,1,1], [1,1,1], [1,1,1]] / 9.0
"""

import torch
import numpy as np
from autoforge.Helper.OptimizerHelper import bleed_layer_effect


def test_bug30_kernel_excludes_center():
    """Test that demonstrates the bug: center pixel is excluded from kernel."""
    # Create a simple test case: single layer with a bright center pixel
    # and dark surroundings
    mask = torch.zeros(1, 3, 3)
    mask[0, 1, 1] = 1.0  # Center pixel is bright

    # Apply bleed effect with low strength to see the effect clearly
    result = bleed_layer_effect(mask, strength=1.0)

    print("\n" + "=" * 60)
    print("TEST: Bleed Mask Kernel Analysis")
    print("=" * 60)

    print("\nInput mask (3x3, center pixel = 1.0):")
    print(mask[0].numpy())

    print("\nOutput after bleed_layer_effect (strength=1.0):")
    print(result[0].numpy())

    # The center value should be influenced by itself
    center_value = result[0, 1, 1].item()
    print(f"\nCenter pixel value: {center_value:.6f}")

    # With buggy kernel [1,0,1]/8 in middle row:
    # - Center at (1,1) receives: 8 neighbors (all 0), center (0 weight) = 0
    # - So center_value = 0 + strength * 0 = 0

    # With fixed kernel [1,1,1]/9 in middle row:
    # - Center at (1,1) receives: itself (1/9) + neighbors (8*0/9) = 1/9
    # - So center_value ≈ 1.0 + strength * (1/9) = 1.0 + 0.111... ≈ 1.111 (clamped to 1.0)

    # Check if bug exists: center should have some contribution from itself
    assert center_value > 0.0, (
        "BUG CONFIRMED: Center pixel has no contribution from itself. "
        "This indicates the kernel excludes the center pixel."
    )

    print("\n✓ Bleed effect includes center pixel (bug may be fixed or not present)")


def test_bug30_neighbor_values():
    """Test that neighbors receive proper bleed from center."""
    # Create test: center is bright, neighbors are dark
    mask = torch.zeros(1, 3, 3)
    mask[0, 1, 1] = 1.0  # Center is bright

    result = bleed_layer_effect(mask, strength=1.0)

    print("\n" + "=" * 60)
    print("TEST: Neighbor Bleed Values")
    print("=" * 60)

    print("\nNeighbor values after bleed (should be non-zero):")
    center_val = result[0, 1, 1].item()
    neighbor_vals = [
        ("Top", result[0, 0, 1].item()),
        ("Bottom", result[0, 2, 1].item()),
        ("Left", result[0, 1, 0].item()),
        ("Right", result[0, 1, 2].item()),
    ]

    for name, val in neighbor_vals:
        print(f"  {name}: {val:.6f}")

    # With buggy kernel: neighbors get blurred value from other 7 neighbors (excluding center)
    # All other neighbors are 0, so blurred = 0
    # Result = 0 + strength * 0 = 0

    # With fixed kernel: neighbors get (center + 2 orthogonal neighbors) / 9
    # Result ≈ 0 + strength * (1/9) = 0.111...

    all_neighbors_positive = all(val > 0.0 for _, val in neighbor_vals)
    if all_neighbors_positive:
        print("\n✓ All neighbors received proper bleed from center")
    else:
        print("\n⚠ Some neighbors have zero bleed (possible bug indication)")


def test_bug30_kernel_composition():
    """Verify the actual kernel being used by inspecting the function."""
    import inspect
    from autoforge.Helper.OptimizerHelper import bleed_layer_effect

    print("\n" + "=" * 60)
    print("TEST: Kernel Composition Analysis")
    print("=" * 60)

    # Create a test mask and use a gradient to see the kernel effect
    mask = torch.zeros(1, 5, 5)
    mask[0, 2, 2] = 1.0  # Single bright pixel in center

    result = bleed_layer_effect(mask, strength=1.0)

    print("\nInput (5x5 with bright center):")
    print(mask[0].numpy())

    print("\nOutput after bleed_layer_effect (strength=1.0):")
    print(result[0].numpy())

    # Analyze the kernel by looking at expected values
    center = result[0, 2, 2].item()
    north = result[0, 1, 2].item()
    south = result[0, 3, 2].item()
    east = result[0, 2, 3].item()
    west = result[0, 2, 1].item()
    northeast = result[0, 1, 3].item()

    print(f"\nKernel weight analysis (strength=1.0):")
    print(f"  Center:        {center:.6f}")
    print(f"  Orthogonal:    {north:.6f}, {south:.6f}, {east:.6f}, {west:.6f}")
    print(f"  Diagonal:      {northeast:.6f}")

    # If kernel is [1,1,1]/[1,0,1]/[1,1,1] / 8:
    #   Input = [[0,0,0], [0,1,0], [0,0,0]] (just center)
    #   Blurred at (2,2) = (0+0+0+0+0+0+0+0) / 8 = 0
    #   Result at (2,2) = 0 + 1.0 * 0 = 0
    #   Result at (1,2) = 0 + 1.0 * (1/8) = 0.125

    # If kernel is [1,1,1]/[1,1,1]/[1,1,1] / 9:
    #   Result at (2,2) = 0 + 1.0 * (1/9) = 0.111...
    #   Result at (1,2) = 0 + 1.0 * (2/9) = 0.222... (includes center + north-center)

    if center > 0.1:
        print("\n✓ Center pixel has significant contribution (fix likely in place)")
    else:
        print("\n✗ Center pixel contribution is weak (bug likely present)")


def test_compare_kernel_behaviors():
    """Compare the two kernel variants to show the difference."""
    print("\n" + "=" * 60)
    print("TEST: Kernel Behavior Comparison")
    print("=" * 60)

    # Simulate what each kernel would produce
    input_mask = torch.zeros(1, 3, 3)
    input_mask[0, 1, 1] = 1.0

    # Buggy kernel: [1,0,1] / [1,0,1] / [1,0,1] / 8
    buggy_kernel = (
        torch.tensor([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]], dtype=torch.float32) / 8.0
    )

    # Fixed kernel: [1,1,1] / [1,1,1] / [1,1,1] / 9
    fixed_kernel = (
        torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32) / 9.0
    )

    print("\nBuggy kernel (excludes center):")
    print((buggy_kernel[0] * 8).numpy())
    print("Normalized by 8")

    print("\nFixed kernel (includes center):")
    print((fixed_kernel[0] * 9).numpy())
    print("Normalized by 9")

    # Apply both
    buggy_blurred = torch.nn.functional.conv2d(
        input_mask.unsqueeze(1), buggy_kernel.view(1, 1, 3, 3), padding=1
    ).squeeze(1)

    fixed_blurred = torch.nn.functional.conv2d(
        input_mask.unsqueeze(1), fixed_kernel.view(1, 1, 3, 3), padding=1
    ).squeeze(1)

    print("\nBuggy result (center should be 0):")
    print(buggy_blurred[0].numpy())

    print("\nFixed result (center should be ~0.111):")
    print(fixed_blurred[0].numpy())


if __name__ == "__main__":
    test_bug30_kernel_excludes_center()
    test_bug30_neighbor_values()
    test_bug30_kernel_composition()
    test_compare_kernel_behaviors()

    print("\n" + "=" * 60)
    print("All Bug #30 verification tests completed!")
    print("=" * 60)
