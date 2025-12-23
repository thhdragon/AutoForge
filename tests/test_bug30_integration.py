"""
Integration test for Bug #30 fix: Verify edge softening improvement
"""

import torch
import numpy as np
from autoforge.Helper.OptimizerHelper import bleed_layer_effect


def test_bleed_strength_effect():
    """
    Test that the bleed effect properly softens hard edges.
    The fixed kernel (including center) should produce smoother transitions.
    """
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Edge Softening Quality")
    print("=" * 60)

    # Create a sharp layer boundary: top half bright, bottom half dark
    mask = torch.zeros(1, 5, 5)
    mask[0, :2, :] = 1.0  # Top 2 rows bright

    print("\nInput mask (sharp edge at row 2):")
    print(mask[0].numpy())

    # Apply with different strength values
    for strength in [0.1, 0.5, 1.0]:
        result = bleed_layer_effect(mask, strength=strength)
        print(f"\nBleed with strength={strength}:")
        print(result[0].numpy())

        # Check that edge is softened: value at boundary should be between 0 and 1
        boundary_value = result[0, 2, 2].item()
        print(f"  Boundary pixel value: {boundary_value:.4f}")

        if 0 < boundary_value < 1:
            print(f"  ✓ Edge properly softened")
        else:
            print(f"  ✗ Edge NOT softened")


def test_layer_interaction():
    """
    Test multi-layer bleed to ensure layers properly interact.
    """
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Multi-Layer Bleed Interaction")
    print("=" * 60)

    # Create 3 layers: pattern like [1, 0.5, 0]
    mask = torch.zeros(3, 3, 3)
    mask[0, :, :] = 1.0
    mask[1, :, :] = 0.5
    # mask[2] stays 0

    result = bleed_layer_effect(mask, strength=0.5)

    print("\nInput layers:")
    for i in range(3):
        print(f"Layer {i}:")
        print(mask[i].numpy())

    print("\nOutput after bleed (strength=0.5):")
    for i in range(3):
        print(f"Layer {i}:")
        print(result[i].numpy())

    # Layer 2 (all zeros) should receive some value from neighbors via bleed
    layer2_max = result[2].max().item()
    print(f"\nLayer 2 maximum value after bleed: {layer2_max:.4f}")

    if layer2_max > 0:
        print("✓ Bleed successfully propagates between layers")
    else:
        print("✗ Bleed did not propagate to empty layer")


def test_strength_scaling():
    """
    Verify that strength parameter properly scales the bleed effect.
    """
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Strength Parameter Scaling")
    print("=" * 60)

    mask = torch.zeros(1, 3, 3)
    mask[0, 1, 1] = 1.0

    results = {}
    for strength in [0.0, 0.1, 0.5, 1.0]:
        result = bleed_layer_effect(mask, strength=strength)
        neighbor_val = result[0, 0, 1].item()  # Top neighbor
        results[strength] = neighbor_val

    print("\nNeighbor values for different strength values:")
    for strength, val in results.items():
        print(f"  strength={strength}: neighbor={val:.6f}")

    # Verify monotonic increase
    strengths = sorted(results.keys())
    values = [results[s] for s in strengths]

    is_monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    if is_monotonic:
        print("\n✓ Bleed strength scales monotonically")
    else:
        print("\n✗ Bleed strength does NOT scale properly")

    # Strength=0 should give original mask
    strength_0_val = results[0.0]
    if strength_0_val < 1e-6:
        print("✓ Strength=0 gives original mask")
    else:
        print("✗ Strength=0 does not give original mask")


if __name__ == "__main__":
    test_bleed_strength_effect()
    test_layer_interaction()
    test_strength_scaling()

    print("\n" + "=" * 60)
    print("All Bug #30 integration tests completed!")
    print("=" * 60)
