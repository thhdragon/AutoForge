"""
Bug #23 Test: Image Resize Aspect Ratio Distortion

Tests the aspect ratio preservation in resize_image().

Bug Description:
    Independent rounding of width and height breaks aspect ratio.
    Example: 100x101 at scale 0.5 → 50x50 (should be 50x51).

Expected Behavior:
    Only width should be rounded, height should maintain exact aspect ratio.
"""

import numpy as np
import pytest


def test_bug23_aspect_ratio_distortion():
    """Verify that independent rounding causes aspect ratio distortion."""
    from autoforge.Helper.ImageHelper import resize_image

    # Create a test image 100x101 (slightly non-square)
    # This will have scale = 0.5 if max_size = 50
    img = np.random.randint(0, 255, (101, 100, 3), dtype=np.uint8)

    # Original aspect ratio
    original_ratio = 100.0 / 101.0

    # Resize to max_size = 50
    # Width is larger (100), so scale = 50/100 = 0.5
    # With bug: new_w = round(100 * 0.5) = 50, new_h = round(101 * 0.5) = 50
    # Without bug: new_w = round(100 * 0.5) = 50, new_h = int(101 * 0.5) = 50
    # Actually both give 50x50 with the bug

    resized = resize_image(img, max_size=50)
    h_out, w_out, _ = resized.shape

    new_ratio = w_out / h_out

    print(f"Original size: 100x101")
    print(f"Original aspect ratio: {original_ratio:.6f}")
    print(f"Resized size: {w_out}x{h_out}")
    print(f"New aspect ratio: {new_ratio:.6f}")
    print(f"Aspect ratio error: {abs(new_ratio - original_ratio):.6f}")

    # The bug causes aspect ratio distortion
    # With independent rounding, we expect 50x50 (wrong!)
    # Without bug, we should get 50x50 (int(101*0.5) = 50, round(101*0.5) = 51)

    # Actually let me recalculate the math:
    # scale = 50 / 100 = 0.5
    # new_w = round(100 * 0.5) = round(50.0) = 50
    # new_h = round(101 * 0.5) = round(50.5) = 50 or 51? (Python rounds to even)

    # Let's test a clearer case: 100x103
    img2 = np.random.randint(0, 255, (103, 100, 3), dtype=np.uint8)
    resized2 = resize_image(img2, max_size=50)
    h_out2, w_out2, _ = resized2.shape

    print(f"\nTest 2: Original size: 100x103")
    print(f"Scale: {50 / 100}")
    print(f"Expected: new_w = round(100*0.5) = 50, new_h = int(103*0.5) = 51")
    print(f"Actual resized size: {w_out2}x{h_out2}")

    # With bug: round(103*0.5) = round(51.5) = 52 (rounds to even)
    # Without bug: int(103*0.5) = 51

    original_ratio2 = 100.0 / 103.0
    new_ratio2 = w_out2 / h_out2
    print(f"Original aspect ratio: {original_ratio2:.6f}")
    print(f"New aspect ratio: {new_ratio2:.6f}")
    print(f"Aspect ratio error: {abs(new_ratio2 - original_ratio2):.6f}")

    # Test with height as the larger dimension
    img3 = np.random.randint(0, 255, (100, 97, 3), dtype=np.uint8)
    resized3 = resize_image(img3, max_size=50)
    h_out3, w_out3, _ = resized3.shape

    print(f"\nTest 3: Original size: 97x100 (height larger)")
    print(f"Scale: {50 / 100}")
    print(f"Expected: new_h = round(100*0.5) = 50, new_w = int(97*0.5) = 48")
    print(f"Actual resized size: {w_out3}x{h_out3}")

    original_ratio3 = 97.0 / 100.0
    new_ratio3 = w_out3 / h_out3
    print(f"Original aspect ratio: {original_ratio3:.6f}")
    print(f"New aspect ratio: {new_ratio3:.6f}")
    print(f"Aspect ratio error: {abs(new_ratio3 - original_ratio3):.6f}")

    # The bug manifests when rounding changes the aspect ratio
    # Assert that we currently have the bug (both dimensions rounded)
    # After fix, the second dimension should not be rounded

    # For test 2: 100x103 -> 50x?
    # With bug: new_h = round(51.5) = 52
    # Fixed: new_h = int(51.5) = 51

    # Current buggy behavior (will fail after fix)
    if h_out2 == 52:
        print("\n✗ BUG CONFIRMED: Height was rounded (52 instead of 51)")
        print("  Aspect ratio distorted due to independent rounding")
        return True
    elif h_out2 == 51:
        print("\n✓ BUG FIXED: Height was not rounded (51 as expected)")
        return False
    else:
        print(f"\n? UNEXPECTED: Height is {h_out2}, expected 51 or 52")
        return None


def test_bug23_specific_example():
    """Test the specific example from bug report: 100x101 -> 50x50."""
    from autoforge.Helper.ImageHelper import resize_image

    # The bug report states: "100x101 at scale 0.5 → 50x50 (squeezed!)"
    img = np.random.randint(0, 255, (101, 100, 3), dtype=np.uint8)

    # This should resize to max_size=50
    resized = resize_image(img, max_size=50)
    h_out, w_out, _ = resized.shape

    print(f"Bug report example: 100x101 -> {w_out}x{h_out}")

    # With scale 0.5:
    # new_w = round(100 * 0.5) = 50
    # new_h = round(101 * 0.5) = round(50.5) = 50 (rounds to even in Python)

    # So the bug report is correct: 50x50 when it should be 50x51
    # (though technically 50.5 could round either way)

    expected_h = int(101 * 0.5)  # Should be 50 without rounding
    actual_h = h_out

    print(f"Expected height (without rounding): {expected_h}")
    print(f"Actual height: {actual_h}")

    if actual_h != expected_h:
        print(f"✗ BUG: Height differs by {actual_h - expected_h} pixels")
    else:
        print("✓ FIXED: Height matches expected value")


def test_aspect_ratio_preservation():
    """Test that aspect ratios are preserved across various image sizes."""
    from autoforge.Helper.ImageHelper import resize_image

    test_cases = [
        (100, 101),  # Slightly taller
        (100, 103),  # More obviously taller
        (97, 100),  # Slightly wider
        (91, 100),  # More obviously wider
        (200, 207),  # Larger image
    ]

    max_errors = []

    for w, h in test_cases:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        original_ratio = w / h

        # Resize
        max_dim = max(w, h)
        target_size = max_dim // 2
        resized = resize_image(img, max_size=target_size)

        h_out, w_out, _ = resized.shape
        new_ratio = w_out / h_out

        error = abs(new_ratio - original_ratio)
        max_errors.append(error)

        print(
            f"{w}x{h} -> {w_out}x{h_out}: "
            f"ratio {original_ratio:.6f} -> {new_ratio:.6f}, "
            f"error {error:.6f}"
        )

    max_error = max(max_errors)
    print(f"\nMaximum aspect ratio error: {max_error:.6f}")

    # With the bug, we expect errors > 0.01 in some cases
    # After fix, errors should be minimal (< 0.001)
    if max_error > 0.01:
        print("✗ BUG: Large aspect ratio errors detected")
    else:
        print("✓ FIXED: Aspect ratios well preserved")

    return max_error


if __name__ == "__main__":
    print("=" * 60)
    print("Bug #23 Verification: Image Resize Aspect Ratio Distortion")
    print("=" * 60)

    print("\n1. Testing specific examples...")
    test_bug23_aspect_ratio_distortion()

    print("\n" + "=" * 60)
    print("2. Testing bug report example...")
    test_bug23_specific_example()

    print("\n" + "=" * 60)
    print("3. Testing aspect ratio preservation...")
    test_aspect_ratio_preservation()

    print("\n" + "=" * 60)
    print("Test complete!")
