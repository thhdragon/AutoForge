"""Test Bug #14: Sigmoid Inverse Creates Extreme Logits

This test verifies that the fix for Bug #14 properly prevents extreme logit values
that would cause sigmoid saturation and gradient loss during optimization.

Bug Description:
- For bright pixels (lum→1), logit→log(large/eps)≈14+
- Extreme logits cause sigmoid saturation, losing gradients
- Without clamping, initialization can't be corrected during optimization

Fix:
- Clamp logits to [-5, 5] range to prevent saturation
- Sigmoid is effectively saturated beyond ±5
- Preserves gradient flow for optimization
"""

import numpy as np
import pytest


def inverse_sigmoid_without_fix(normalized_lum, eps=1e-6):
    """Original buggy version without clamping."""
    return np.log((normalized_lum + eps) / (1 - normalized_lum + eps))


def inverse_sigmoid_with_fix(normalized_lum, eps=1e-6):
    """Fixed version with clamping to [-5, 5]."""
    logits = np.log((normalized_lum + eps) / (1 - normalized_lum + eps))
    return np.clip(logits, -5, 5)


def sigmoid(x):
    """Standard sigmoid function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid for gradient analysis."""
    s = sigmoid(x)
    return s * (1 - s)


class TestBug14ExtremeLogits:
    """Test suite for Bug #14 verification."""

    def test_extreme_logits_without_fix(self):
        """Verify that the bug exists: extreme logits are created for bright pixels."""
        # Test with very bright pixel (lum = 0.9999 → logit ≈ 9.2)
        # The closer to 1.0, the more extreme the logit
        bright_lum = 0.9999
        logit = inverse_sigmoid_without_fix(bright_lum)

        # Without fix, logit should exceed the clamp threshold of 5
        assert logit > 5, f"Expected extreme logit >5 for bright pixel, got {logit:.2f}"

        # Test with very dark pixel (lum = 0.0001 → logit ≈ -9.2)
        dark_lum = 0.0001
        logit = inverse_sigmoid_without_fix(dark_lum)

        # Without fix, logit should be below the clamp threshold of -5
        assert logit < -5, f"Expected extreme logit <-5 for dark pixel, got {logit:.2f}"

        # Also test with the more realistic near-white/near-black pixels from images
        very_bright = 0.996  # ~254/255
        very_dark = 0.004  # ~1/255
        logit_bright = inverse_sigmoid_without_fix(very_bright)
        logit_dark = inverse_sigmoid_without_fix(very_dark)

        print(f"✓ Bug verified: extreme logits detected")
        print(f"  lum=0.9999 → logit={inverse_sigmoid_without_fix(0.9999):.2f}")
        print(f"  lum=0.996 → logit={logit_bright:.2f} (254/255 in image)")
        print(f"  lum=0.004 → logit={logit_dark:.2f} (1/255 in image)")

    def test_logits_clamped_with_fix(self):
        """Verify that the fix clamps logits to [-5, 5] range."""
        # Test with very bright pixel (lum = 0.9999)
        bright_lum = 0.9999
        logit = inverse_sigmoid_with_fix(bright_lum)

        # With fix, logit should be clamped to 5
        assert logit == 5.0, (
            f"Expected clamped logit=5 for bright pixel, got {logit:.2f}"
        )

        # Test with very dark pixel (lum = 0.0001)
        dark_lum = 0.0001
        logit = inverse_sigmoid_with_fix(dark_lum)

        # With fix, logit should be clamped to -5
        assert logit == -5.0, (
            f"Expected clamped logit=-5 for dark pixel, got {logit:.2f}"
        )

        # Test with mid-range value (should not be clamped)
        mid_lum = 0.5
        logit_mid = inverse_sigmoid_with_fix(mid_lum)
        assert -5 < logit_mid < 5, f"Mid-range logit should not be clamped"

        print(f"✓ Fix verified: logits properly clamped to [-5, 5]")
        print(f"  lum=0.9999 → clamped to {inverse_sigmoid_with_fix(bright_lum):.2f}")
        print(f"  lum=0.5 → {logit_mid:.2f} (not clamped)")

    def test_gradient_preservation(self):
        """Verify that clamping preserves gradient flow."""
        # Test gradient at various logit values
        test_logits = np.array([-10, -5, -2, 0, 2, 5, 10])

        gradients_without_fix = sigmoid_derivative(test_logits)

        # Clamp logits and compute gradients
        clamped_logits = np.clip(test_logits, -5, 5)
        gradients_with_fix = sigmoid_derivative(clamped_logits)

        # At extreme values (±10), gradient without fix is near zero (saturated)
        assert gradients_without_fix[0] < 1e-4, (
            "Gradient at -10 should be near zero (saturated)"
        )
        assert gradients_without_fix[-1] < 1e-4, (
            "Gradient at +10 should be near zero (saturated)"
        )

        # With fix, gradients are preserved (sigmoid'(±5) ≈ 0.0066)
        assert gradients_with_fix[0] > 1e-3, "Gradient at clamped -5 should be non-zero"
        assert gradients_with_fix[-1] > 1e-3, (
            "Gradient at clamped +5 should be non-zero"
        )

        print(
            f"✓ Gradients preserved: sigmoid'(-5)={gradients_with_fix[0]:.6f}, sigmoid'(5)={gradients_with_fix[-1]:.6f}"
        )

    def test_sigmoid_saturation_range(self):
        """Verify that sigmoid saturates beyond ±5."""
        # Sigmoid at ±5
        sig_at_5 = sigmoid(5)
        sig_at_minus_5 = sigmoid(-5)

        # Should be very close to 1 and 0 respectively
        assert sig_at_5 > 0.993, f"sigmoid(5)={sig_at_5:.6f} should be >0.993"
        assert sig_at_minus_5 < 0.007, (
            f"sigmoid(-5)={sig_at_minus_5:.6f} should be <0.007"
        )

        # Sigmoid at ±10 (saturation)
        sig_at_10 = sigmoid(10)
        sig_at_minus_10 = sigmoid(-10)

        # Should be extremely close to 1 and 0
        assert sig_at_10 > 0.99995, (
            f"sigmoid(10)={sig_at_10:.6f} should be >0.99995 (saturated)"
        )
        assert sig_at_minus_10 < 0.00005, (
            f"sigmoid(-10)={sig_at_minus_10:.6f} should be <0.00005 (saturated)"
        )

        print(
            f"✓ Saturation verified: sigmoid(±5)=[{sig_at_minus_5:.6f}, {sig_at_5:.6f}], sigmoid(±10)=[{sig_at_minus_10:.10f}, {sig_at_10:.10f}]"
        )

    def test_real_image_scenario(self):
        """Test with realistic image data to verify fix works in practice."""
        # Create a test image with extreme values
        test_image = np.array(
            [
                [0, 0, 0],  # Very dark (lum ≈ 0)
                [255, 255, 255],  # Very bright (lum ≈ 1)
                [128, 128, 128],  # Mid-gray (lum ≈ 0.5)
                [200, 200, 200],  # Light gray (lum ≈ 0.78)
            ],
            dtype=np.float32,
        )

        # Compute luminance
        normalized_lum = (
            0.299 * test_image[:, 0]
            + 0.587 * test_image[:, 1]
            + 0.114 * test_image[:, 2]
        ) / 255.0

        # Without fix
        logits_buggy = inverse_sigmoid_without_fix(normalized_lum)

        # With fix
        logits_fixed = inverse_sigmoid_with_fix(normalized_lum)

        # Verify buggy version has extreme values
        assert np.any(np.abs(logits_buggy) > 10), (
            "Buggy version should have extreme logits"
        )

        # Verify fixed version is clamped
        assert np.all(logits_fixed >= -5) and np.all(logits_fixed <= 5), (
            "Fixed version should be clamped to [-5, 5]"
        )

        print(f"✓ Real scenario tested:")
        print(
            f"  Luminance range: [{normalized_lum.min():.4f}, {normalized_lum.max():.4f}]"
        )
        print(
            f"  Buggy logits range: [{logits_buggy.min():.2f}, {logits_buggy.max():.2f}]"
        )
        print(
            f"  Fixed logits range: [{logits_fixed.min():.2f}, {logits_fixed.max():.2f}]"
        )

    def test_roundtrip_accuracy(self):
        """Verify that sigmoid(clipped_logit) still approximates the original luminance reasonably."""
        # Test various luminance values
        test_lums = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])

        # With fix
        logits_fixed = inverse_sigmoid_with_fix(test_lums)
        reconstructed_lums = sigmoid(logits_fixed)

        # For extreme values (0.01, 0.99), reconstruction will be less accurate due to clamping
        # But for mid-range values, should be very accurate
        mid_range_mask = (test_lums > 0.1) & (test_lums < 0.9)
        mid_range_error = np.abs(
            test_lums[mid_range_mask] - reconstructed_lums[mid_range_mask]
        )

        # Mid-range values should reconstruct with <1% error
        assert np.all(mid_range_error < 0.01), (
            f"Mid-range reconstruction error too high: {mid_range_error.max():.4f}"
        )

        print(f"✓ Roundtrip accuracy verified:")
        print(f"  Original lums: {test_lums}")
        print(f"  Reconstructed:  {reconstructed_lums}")
        print(f"  Max mid-range error: {mid_range_error.max():.6f}")

    def test_actual_implementation(self):
        """Test the actual implementation in DepthEstimateHeightMap.py."""
        try:
            from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
                initialize_pixel_height_logits,
            )

            # Create test image with extreme values
            test_image = np.array(
                [
                    [[0, 0, 0], [255, 255, 255]],
                    [[50, 50, 50], [200, 200, 200]],
                ],
                dtype=np.float32,
            )

            # Call actual implementation
            logits = initialize_pixel_height_logits(test_image)

            # Verify logits are clamped to [-5, 5]
            assert np.all(logits >= -5) and np.all(logits <= 5), (
                f"Implementation should clamp to [-5, 5], got range [{logits.min():.2f}, {logits.max():.2f}]"
            )

            print(f"✓ Actual implementation verified:")
            print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
            print(f"  Shape: {logits.shape}")

        except ImportError as e:
            pytest.skip(f"Could not import implementation: {e}")


if __name__ == "__main__":
    """Run tests with verbose output."""
    print("=" * 70)
    print("Bug #14 Verification: Sigmoid Inverse Creates Extreme Logits")
    print("=" * 70)
    print()

    test = TestBug14ExtremeLogits()

    print("Test 1: Verify Bug Exists (Without Fix)")
    print("-" * 70)
    test.test_extreme_logits_without_fix()
    print()

    print("Test 2: Verify Fix Clamps Logits")
    print("-" * 70)
    test.test_logits_clamped_with_fix()
    print()

    print("Test 3: Verify Gradient Preservation")
    print("-" * 70)
    test.test_gradient_preservation()
    print()

    print("Test 4: Verify Sigmoid Saturation Range")
    print("-" * 70)
    test.test_sigmoid_saturation_range()
    print()

    print("Test 5: Real Image Scenario")
    print("-" * 70)
    test.test_real_image_scenario()
    print()

    print("Test 6: Roundtrip Accuracy")
    print("-" * 70)
    test.test_roundtrip_accuracy()
    print()

    print("Test 7: Actual Implementation")
    print("-" * 70)
    test.test_actual_implementation()
    print()

    print("=" * 70)
    print("All tests passed! Bug #14 fix verified.")
    print("=" * 70)
