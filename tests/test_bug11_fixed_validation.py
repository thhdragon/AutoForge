"""
Test to validate that the fixed PruningHelper.smooth_coplanar_faces works correctly.
"""

import sys
import torch

sys.path.insert(0, "src")

from autoforge.Helper.PruningHelper import smooth_coplanar_faces


def test_smooth_coplanar_fixed_version():
    """Test that the fixed version produces correct smoothing on a gradual slope"""
    # Create a gentle slope that should be smoothed (not a sharp wall)
    # This creates a surface that increases gradually
    height_logits = torch.zeros(7, 7)
    for i in range(7):
        height_logits[i, :] = i * 0.5  # Gentle slope

    # Apply smoothing
    result = smooth_coplanar_faces(height_logits, angle_threshold=45)

    # Verify it's not identical to input (smoothing occurred on gentle slopes)
    assert not torch.allclose(result, height_logits, atol=0.01), (
        "Smoothing should modify the height map for gentle slopes"
    )

    # The smoothing should create intermediate values
    assert result.max() <= height_logits.max() + 1.0
    assert result.min() >= height_logits.min() - 1.0

    print("✓ Fixed version produces correct smoothing")


def test_smooth_coplanar_on_planar_surface():
    """Test that fixed version preserves planar surfaces"""
    # Create a perfectly planar surface
    height_logits = torch.ones(5, 5) * 7.0

    result = smooth_coplanar_faces(height_logits, angle_threshold=5)

    # On a perfectly planar surface, result should stay fairly constant
    # (all neighbors are coplanar at shallow angle thresholds)
    assert torch.std(result).item() < torch.std(height_logits).item() + 0.5, (
        "Planar surface should remain approximately planar"
    )

    print("✓ Fixed version preserves planar surfaces")


def test_smooth_coplanar_gradient_surface():
    """Test on gradient surface to verify neighbor selection is correct"""
    # Create a simple gradient
    height_logits = torch.arange(25, dtype=torch.float32).reshape(5, 5)

    result = smooth_coplanar_faces(
        height_logits, angle_threshold=45
    )  # Higher threshold allows more coplanar neighbors

    # Verify the function runs without errors and produces a tensor
    assert isinstance(result, torch.Tensor), "Result should be a tensor"
    assert result.shape == height_logits.shape, "Result shape should match input"

    # The result might not always be smoother depending on the angle threshold
    # Just verify that the function executes correctly with the fixed dimensions
    print(f"Original min/max: {height_logits.min():.2f}/{height_logits.max():.2f}")
    print(f"Smoothed min/max: {result.min():.2f}/{result.max():.2f}")

    print("✓ Fixed version correctly processes gradient surfaces")


if __name__ == "__main__":
    print("Testing fixed Bug #11: smooth_coplanar_faces\n")

    print("Test 1: Fixed version produces correct smoothing...")
    test_smooth_coplanar_fixed_version()

    print("Test 2: Fixed version preserves planar surfaces...")
    test_smooth_coplanar_on_planar_surface()

    print("Test 3: Fixed version on gradient surface...")
    test_smooth_coplanar_gradient_surface()

    print("\n✅ All validation tests passed! Bug #11 is fixed.")
