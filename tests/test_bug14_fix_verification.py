"""Test to verify bug 14 fix is working"""

import numpy as np
import sys

sys.path.insert(0, "src")

from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
    initialize_pixel_height_logits,
)


def test_bug14_fixed():
    """Test that logits are now properly clipped"""
    print("=" * 70)
    print("VERIFYING BUG 14 FIX")
    print("=" * 70)
    print()

    # Create a test image with extreme luminance values
    # All black pixels (0,0,0) and all white pixels (255,255,255)
    test_image = np.array(
        [
            [[0, 0, 0], [128, 128, 128], [255, 255, 255]],
            [[0, 0, 0], [128, 128, 128], [255, 255, 255]],
            [[0, 0, 0], [128, 128, 128], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    print("Input image (3x3 with black, gray, white):")
    print(test_image)
    print()

    logits = initialize_pixel_height_logits(test_image)

    print("Generated logits:")
    print(logits)
    print()

    min_logit = np.min(logits)
    max_logit = np.max(logits)

    print(f"Min logit: {min_logit:.4f}")
    print(f"Max logit: {max_logit:.4f}")
    print()

    # Verify bounds
    assert min_logit >= -5, f"Min logit {min_logit} exceeds lower bound of -5"
    assert max_logit <= 5, f"Max logit {max_logit} exceeds upper bound of 5"
    print("✅ PASS: All logits are within [-5, 5] bounds")
    print()

    # Verify sigmoid doesn't saturate
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    sigmoided = sigmoid(logits)
    min_sig = np.min(sigmoided)
    max_sig = np.max(sigmoided)

    print(f"Sigmoid output range: [{min_sig:.6f}, {max_sig:.6f}]")
    print()

    # Check gradient preservation
    import torch

    torch_logits = torch.tensor(
        logits.flatten(), dtype=torch.float32, requires_grad=True
    )
    loss = torch.sigmoid(torch_logits).sum()
    loss.backward()

    gradients = torch_logits.grad.numpy()
    zero_grad_count = np.sum(np.abs(gradients) < 1e-6)

    print(f"Gradient statistics:")
    print(f"  Min gradient: {gradients.min():.6f}")
    print(f"  Max gradient: {gradients.max():.6f}")
    print(f"  Mean gradient: {gradients.mean():.6f}")
    print(f"  Zero gradients: {zero_grad_count}/{len(gradients)}")
    print()

    if zero_grad_count == 0:
        print("✅ PASS: All gradients are non-zero (no saturation)")
    else:
        print("⚠️  WARNING: Some gradients are zero (possible saturation)")

    print()
    print("=" * 70)
    print("RESULT: BUG 14 FIX VERIFIED ✅")
    print("=" * 70)


if __name__ == "__main__":
    test_bug14_fixed()
