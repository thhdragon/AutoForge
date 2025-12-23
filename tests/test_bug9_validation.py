#!/usr/bin/env python
"""Test that bug #9 validation works correctly."""
from __future__ import annotations

import types

import numpy as np
import torch

from autoforge.Modules.Optimizer import FilamentOptimizer


def test_tau_validation_fails() -> bool | None:
    """Test that init_tau < final_tau raises ValueError."""
    args = types.SimpleNamespace(
        iterations=2,
        warmup_fraction=0.0,
        learning_rate_warmup_fraction=0.0,
        init_tau=0.01,  # WRONG: init < final
        final_tau=1.0,  # WRONG: final > init
        learning_rate=0.01,
        layer_height=0.04,
        max_layers=4,
        visualize=False,
        tensorboard=False,
        disable_visualization_for_gradio=1,
        output_folder="/tmp",
    )

    device = torch.device("cpu")
    H, W = 16, 16
    target = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8).to(torch.float32)
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.zeros((4, 3), dtype=np.float32)
    material_colors = torch.rand(3, 3)
    material_TDs = torch.ones(3)
    background = torch.tensor([0.0, 0.0, 0.0])

    try:
        optimizer = FilamentOptimizer(
            args=args,
            target=target,
            pixel_height_logits_init=pixel_height_logits_init,
            pixel_height_labels=pixel_height_labels,
            global_logits_init=global_logits_init,
            material_colors=material_colors,
            material_TDs=material_TDs,
            background=background,
            device=device,
            perception_loss_module=None,
        )
        print("ERROR: Should have raised ValueError!")
        return False
    except ValueError as e:
        print("✓ Validation caught error correctly:")
        print(f"  {e}")
        assert "init_tau" in str(e)
        assert "final_tau" in str(e)
        return True


def test_tau_validation_passes():
    """Test that init_tau >= final_tau succeeds."""
    args = types.SimpleNamespace(
        iterations=2,
        warmup_fraction=0.0,
        learning_rate_warmup_fraction=0.0,
        init_tau=1.0,  # Correct: init >= final
        final_tau=0.01,  # Correct: final <= init
        learning_rate=0.01,
        layer_height=0.04,
        max_layers=4,
        visualize=False,
        tensorboard=False,
        disable_visualization_for_gradio=1,
        output_folder="/tmp",
    )

    device = torch.device("cpu")
    H, W = 16, 16
    target = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8).to(torch.float32)
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.zeros((4, 3), dtype=np.float32)
    material_colors = torch.rand(3, 3)
    material_TDs = torch.ones(3)
    background = torch.tensor([0.0, 0.0, 0.0])

    try:
        optimizer = FilamentOptimizer(
            args=args,
            target=target,
            pixel_height_logits_init=pixel_height_logits_init,
            pixel_height_labels=pixel_height_labels,
            global_logits_init=global_logits_init,
            material_colors=material_colors,
            material_TDs=material_TDs,
            background=background,
            device=device,
            perception_loss_module=None,
        )
        print("✓ Valid tau values accepted correctly")
        return True
    except ValueError as e:
        print(f"ERROR: Should not have raised ValueError: {e}")
        return False


if __name__ == "__main__":
    print("Testing Bug #9 Tau Validation\n")
    print("=" * 60)
    print("Test 1: init_tau < final_tau (should fail)")
    print("=" * 60)
    result1 = test_tau_validation_fails()

    print("\n" + "=" * 60)
    print("Test 2: init_tau >= final_tau (should pass)")
    print("=" * 60)
    result2 = test_tau_validation_passes()

    print("\n" + "=" * 60)
    if result1 and result2:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
