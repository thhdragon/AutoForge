"""
Test to verify Bug 13: Device Mismatch After Pruning

Bug 13 states that when disc_to_logits() creates a new global_logits tensor
during pruning, it doesn't transfer the tensor to the optimizer's device,
causing device mismatch errors when used in downstream operations.

After the fix, all disc_to_logits assignments should include .to(optimizer.device)
"""

import torch
import numpy as np
from src.autoforge.Helper.PruningHelper import disc_to_logits, prune_num_colors


def test_disc_to_logits_device_mismatch():
    """
    Verify that disc_to_logits creates tensors that might be on CPU
    when device is GPU, or vice versa.
    """
    # Simulate a discrete global assignment
    num_materials = 5
    max_layers = 10
    dg = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)

    # Check which device the tensor ends up on
    logits = disc_to_logits(dg, num_materials, big_pos=1e5)

    print(f"Input dg device: {dg.device}")
    print(f"Output logits device: {logits.device}")

    # Test on different devices
    if torch.cuda.is_available():
        print("\n--- Testing with CUDA ---")
        dg_gpu = dg.to("cuda")
        logits_gpu = disc_to_logits(dg_gpu, num_materials, big_pos=1e5)
        print(f"Input dg_gpu device: {dg_gpu.device}")
        print(f"Output logits_gpu device: {logits_gpu.device}")

        # Now test the actual problem: assigning to a params dict
        # and then trying to use it with tensors on a different device
        device_gpu = torch.device("cuda")
        device_cpu = torch.device("cpu")

        # Simulate optimizer.best_params with tensors on GPU
        best_params_gpu = {
            "global_logits": logits_gpu,
            "pixel_height_logits": torch.randn(256, 256, device=device_gpu),
        }

        # Now simulate pruning creating new logits on CPU (the bug)
        dg_pruned = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=torch.long)
        logits_pruned = disc_to_logits(dg_pruned, num_materials, big_pos=1e5)

        print(f"\nAfter pruning:")
        print(f"Pruned logits device: {logits_pruned.device}")
        print(
            f"Best params global_logits device: {best_params_gpu['global_logits'].device}"
        )

        # Check for device mismatch
        if logits_pruned.device != best_params_gpu["pixel_height_logits"].device:
            print(f"\n⚠️  DEVICE MISMATCH DETECTED!")
            print(f"   New global_logits on {logits_pruned.device}")
            print(
                f"   But pixel_height_logits on {best_params_gpu['pixel_height_logits'].device}"
            )

            # Try to use them together (this would fail)
            try:
                # This simulates what happens in loss computation
                combined = logits_pruned + best_params_gpu["pixel_height_logits"]
                print("ERROR: Should have failed due to device mismatch!")
            except RuntimeError as e:
                print(f"   Expected error: {e}")
                return True
        else:
            print("\nNo device mismatch (both on same device)")
            return False

    return None


if __name__ == "__main__":
    result = test_disc_to_logits_device_mismatch()
    if result:
        print("\n✓ Bug 13 was NOT fixed (device mismatch still exists)")
    elif result is False:
        print("\n✓ Bug 13 FIXED: No device mismatch detected")
    else:
        print("\n? No GPU available for testing")

    # Now check if the fix was applied to PruningHelper
    print("\n--- Checking if fix was applied to PruningHelper.py ---")
    with open("src/autoforge/Helper/PruningHelper.py", "r") as f:
        content = f.read()

    # Count occurrences of the fix pattern
    fix_pattern = "disc_to_logits(\n    best_dg"
    fix_with_device = "disc_to_logits(\n    best_dg"

    # Count assignments with .to(optimizer.device)
    fixed_count = content.count(
        "disc_to_logits(\n        best_dg, num_materials=num_materials, big_pos=1e5\n    ).to(optimizer.device)"
    )
    fixed_count += content.count(
        "disc_to_logits(\n    best_dg, num_materials=num_materials, big_pos=1e5\n).to(optimizer.device)"
    )

    print(f"Found {fixed_count} fixed assignments with .to(optimizer.device)")
    if fixed_count >= 5:
        print("✓ All 5 critical locations have been fixed!")
    else:
        print(f"⚠️  Only {fixed_count} of 5 locations fixed")
