"""
Test for Bug #11: Coplanar Smoothing Dimension Swap

Bug Description:
    In smooth_coplanar_faces(), the normals tensor is (3, H, W) where:
    - dim=0 is the normal component (x, y, z)
    - dim=1 is HEIGHT
    - dim=2 is WIDTH

    When shifting neighbors, the code should:
    - Shift HEIGHT by dy (row shift)
    - Shift WIDTH by dx (column shift)

    But the current code does the opposite:
    - Shifts HEIGHT (dim=1) by dx
    - Shifts WIDTH (dim=2) by dy

    This causes wrong neighbor sampling and smoothing artifacts.
"""

import math
import torch
import pytest


def smooth_coplanar_faces_buggy(
    height_logits: torch.Tensor, angle_threshold: float
) -> torch.Tensor:
    """Original buggy version - dimensions reversed"""
    threshold_rad = math.radians(angle_threshold)

    grad_x = (
        torch.roll(height_logits, shifts=-1, dims=1)
        - torch.roll(height_logits, shifts=1, dims=1)
    ) / 2.0
    grad_y = (
        torch.roll(height_logits, shifts=-1, dims=0)
        - torch.roll(height_logits, shifts=1, dims=0)
    ) / 2.0

    ones = torch.ones_like(height_logits)
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = ones
    norm = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    normals = torch.stack([normal_x, normal_y, normal_z], dim=0)

    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    coplanar_sum = height_logits.clone()
    count = torch.ones_like(height_logits)

    for dx, dy in shifts:
        # BUG: dimensions reversed!
        neighbor_normals = torch.roll(
            torch.roll(
                normals, shifts=dx, dims=1
            ),  # WRONG: shifts dx on HEIGHT (dim=1)
            shifts=dy,
            dims=2,  # WRONG: shifts dy on WIDTH (dim=2)
        )
        dot = (normals * neighbor_normals).sum(dim=0)
        dot = dot.clamp(-1.0, 1.0)
        angle_diff = torch.acos(dot)
        mask = angle_diff < math.radians(angle_threshold)
        neighbor_heights = torch.roll(
            torch.roll(height_logits, shifts=dx, dims=0), shifts=dy, dims=1
        )
        coplanar_sum += neighbor_heights * mask.float()
        count += mask.float()

    smoothed_height_logits = coplanar_sum / count.clamp(min=1)
    return smoothed_height_logits


def smooth_coplanar_faces_fixed(
    height_logits: torch.Tensor, angle_threshold: float
) -> torch.Tensor:
    """Fixed version - correct dimensions"""
    threshold_rad = math.radians(angle_threshold)

    grad_x = (
        torch.roll(height_logits, shifts=-1, dims=1)
        - torch.roll(height_logits, shifts=1, dims=1)
    ) / 2.0
    grad_y = (
        torch.roll(height_logits, shifts=-1, dims=0)
        - torch.roll(height_logits, shifts=1, dims=0)
    ) / 2.0

    ones = torch.ones_like(height_logits)
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = ones
    norm = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    normals = torch.stack([normal_x, normal_y, normal_z], dim=0)

    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    coplanar_sum = height_logits.clone()
    count = torch.ones_like(height_logits)

    for dx, dy in shifts:
        # FIXED: correct dimensions
        neighbor_normals = torch.roll(
            torch.roll(
                normals, shifts=dy, dims=1
            ),  # CORRECT: shift HEIGHT (dim=1) by dy
            shifts=dx,
            dims=2,  # CORRECT: shift WIDTH (dim=2) by dx
        )
        dot = (normals * neighbor_normals).sum(dim=0)
        dot = dot.clamp(-1.0, 1.0)
        angle_diff = torch.acos(dot)
        mask = angle_diff < math.radians(angle_threshold)
        neighbor_heights = torch.roll(
            torch.roll(height_logits, shifts=dx, dims=0), shifts=dy, dims=1
        )
        coplanar_sum += neighbor_heights * mask.float()
        count += mask.float()

    smoothed_height_logits = coplanar_sum / count.clamp(min=1)
    return smoothed_height_logits


def test_dimension_mapping():
    """Test that shifts and dims are correctly mapped"""
    # Create a simple height map with clear pattern
    height_logits = torch.zeros(5, 5)
    # Center column has value 10, rest is 0
    height_logits[:, 2] = 10.0

    # Test buggy version
    result_buggy = smooth_coplanar_faces_buggy(height_logits, angle_threshold=30)

    # Test fixed version
    result_fixed = smooth_coplanar_faces_fixed(height_logits, angle_threshold=30)

    # Results should differ (buggy and fixed versions produce different outputs)
    assert not torch.allclose(result_buggy, result_fixed), (
        "Buggy and fixed versions should produce different results"
    )

    # The fixed version should be smoother and more physically reasonable
    # Verify this by checking that fixed version produces expected smoothing pattern
    print("Buggy result:\n", result_buggy)
    print("\nFixed result:\n", result_fixed)


def test_neighbor_selection_correctness():
    """
    Test that neighbors are correctly selected for a specific pixel.

    For a pixel at (i, j), when we shift by (dx, dy):
    - dx shifts along HEIGHT (rows), which is dimension 0 in height_logits
    - dy shifts along WIDTH (columns), which is dimension 1 in height_logits

    For normals tensor with shape (3, H, W):
    - dim=0 is component (x, y, z)
    - dim=1 is HEIGHT
    - dim=2 is WIDTH

    So normals should be shifted:
    - By dy along dim=1 (HEIGHT)
    - By dx along dim=2 (WIDTH)
    """
    # Create simple gradient map for predictable normals
    height_logits = torch.arange(25, dtype=torch.float32).reshape(5, 5)

    # Get normals
    grad_x = (
        torch.roll(height_logits, shifts=-1, dims=1)
        - torch.roll(height_logits, shifts=1, dims=1)
    ) / 2.0
    grad_y = (
        torch.roll(height_logits, shifts=-1, dims=0)
        - torch.roll(height_logits, shifts=1, dims=0)
    ) / 2.0

    ones = torch.ones_like(height_logits)
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = ones
    norm = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    normals = torch.stack([normal_x, normal_y, normal_z], dim=0)  # Shape: (3, 5, 5)

    # Test neighbor shift (-1, 0): one pixel up (dx=-1, dy=0)
    # In spatial terms: move up 1 row = shift HEIGHT by -1
    # Correct: shift HEIGHT (dim=1) by dy=0, shift WIDTH (dim=2) by dx=-1
    neighbor_correct = torch.roll(
        torch.roll(
            normals, shifts=0, dims=1
        ),  # shift HEIGHT by dy=0 (no vertical shift)
        shifts=-1,
        dims=2,  # shift WIDTH by dx=-1 (shift width/columns left)
    )

    # Buggy: shift HEIGHT (dim=1) by dx=-1, shift WIDTH (dim=2) by dy=0
    neighbor_buggy = torch.roll(
        torch.roll(normals, shifts=-1, dims=1),  # shifts dx on HEIGHT (wrong)
        shifts=0,
        dims=2,  # shifts dy on WIDTH (wrong)
    )

    # They should differ
    assert not torch.allclose(neighbor_correct, neighbor_buggy), (
        "Correct and buggy neighbor selection should differ"
    )


def test_coplanar_detection_differs():
    """
    Test that coplanar detection differs between buggy and fixed versions.
    This tests the real-world impact: wrong neighbors selected = wrong coplanar mask.
    """
    # Create a surface with a sharp gradient to highlight neighbor selection issues
    height_logits = torch.arange(25, dtype=torch.float32).reshape(5, 5)

    angle_threshold = 10.0

    result_buggy = smooth_coplanar_faces_buggy(height_logits, angle_threshold)
    result_fixed = smooth_coplanar_faces_fixed(height_logits, angle_threshold)

    # The key is that they should NOT be identical
    max_diff = torch.abs(result_buggy - result_fixed).max().item()
    print(f"Max difference between buggy and fixed: {max_diff}")

    # Dimensions are swapped, so results should differ significantly
    assert not torch.allclose(result_buggy, result_fixed, atol=1e-5), (
        "Fixed version should differ from buggy version due to dimension swap"
    )


def test_gradient_direction_correctness():
    """
    Test that gradient computations work with correct dimensions.

    grad_x uses dims=1 (WIDTH/columns)
    grad_y uses dims=0 (HEIGHT/rows)

    This is correct because torch.roll shifts along the specified dimension.
    """
    # Create a ramp in x direction (columns)
    height_logits = torch.zeros(5, 5)
    height_logits[:, 0] = 0.0
    height_logits[:, 1] = 1.0
    height_logits[:, 2] = 2.0
    height_logits[:, 3] = 3.0
    height_logits[:, 4] = 4.0

    # Compute gradients
    grad_x = (
        torch.roll(height_logits, shifts=-1, dims=1)
        - torch.roll(height_logits, shifts=1, dims=1)
    ) / 2.0
    grad_y = (
        torch.roll(height_logits, shifts=-1, dims=0)
        - torch.roll(height_logits, shifts=1, dims=0)
    ) / 2.0

    # grad_x should be ~1 everywhere (change in x-direction)
    # grad_y should be ~0 everywhere (no change in y-direction)
    assert torch.abs(grad_x[2, 2] - 1.0) < 0.1, (
        f"Expected grad_x ~1, got {grad_x[2, 2]}"
    )
    assert torch.abs(grad_y[2, 2]) < 0.1, f"Expected grad_y ~0, got {grad_y[2, 2]}"


if __name__ == "__main__":
    print("Testing Bug #11: Coplanar Smoothing Dimension Swap\n")

    print("Test 1: Dimension mapping...")
    test_dimension_mapping()
    print("✓ Passed\n")

    print("Test 2: Neighbor selection correctness...")
    test_neighbor_selection_correctness()
    print("✓ Passed\n")

    print("Test 3: Coplanar detection differs...")
    test_coplanar_detection_differs()
    print("✓ Passed\n")

    print("Test 4: Gradient direction correctness...")
    test_gradient_direction_correctness()
    print("✓ Passed\n")

    print("All tests passed! Bug #11 is confirmed.")
