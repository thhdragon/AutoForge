"""
Test for Bug #9: Negative Tau Schedule Prevents Annealing

Tests that tau schedule properly handles init_tau < final_tau case
and verifies that tau values decrease monotonically during optimization.
"""

import pytest
import torch
import numpy as np
from argparse import Namespace


def test_negative_decay_rate_bug():
    """
    Bug demonstration: If init_tau < final_tau, decay_rate becomes negative,
    causing tau to increase instead of decrease.
    """
    from autoforge.Modules.Optimizer import FilamentOptimizer

    # Setup scenario where init_tau < final_tau (user mistake)
    args = Namespace(
        init_tau=0.1,  # Smaller initial tau
        final_tau=1.0,  # Larger final tau (mistake!)
        iterations=100,
        warmup_fraction=0.1,
        learning_rate=0.01,
        learning_rate_warmup_fraction=0.1,
        seed=42,
        tensorboard=False,
        max_layers=10,
        layer_height=0.2,
        visualize=False,
        run_name="",
        disable_visualization_for_gradio=1,
        output_folder=".",
    )

    # Create minimal setup
    H, W = 32, 32
    num_materials = 10
    material_colors = torch.rand(num_materials, 3)
    material_TDs = torch.rand(num_materials)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = torch.rand(H, W, 3).float()
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.random.randn(num_materials, 2).astype(np.float32)
    device = torch.device("cpu")

    class DummyLoss(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

    # This should raise a ValueError with the fix
    with pytest.raises(ValueError, match="init_tau.*must be >= final_tau"):
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
            perception_loss_module=DummyLoss(),
        )


def test_tau_decreases_monotonically():
    """
    Verify that tau values decrease monotonically during optimization.
    """
    from autoforge.Modules.Optimizer import FilamentOptimizer

    args = Namespace(
        init_tau=1.0,
        final_tau=0.01,
        iterations=100,
        warmup_fraction=0.1,
        learning_rate=0.01,
        learning_rate_warmup_fraction=0.1,
        seed=42,
        tensorboard=False,
        max_layers=10,
        layer_height=0.2,
        visualize=False,
        run_name="",
        disable_visualization_for_gradio=1,
        output_folder=".",
    )

    # Create minimal setup
    H, W = 32, 32
    num_materials = 10
    material_colors = torch.rand(num_materials, 3)
    material_TDs = torch.rand(num_materials)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = torch.rand(H, W, 3).float()
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.random.randn(num_materials, 2).astype(np.float32)
    device = torch.device("cpu")

    class DummyLoss(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

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
        perception_loss_module=DummyLoss(),
    )

    # Collect tau values throughout optimization
    tau_values = []
    for i in range(args.iterations):
        tau_h, tau_g = optimizer._get_tau()
        tau_values.append(tau_h)
        optimizer.num_steps_done += 1

    # After warmup, tau should decrease monotonically
    warmup_end = int(args.warmup_fraction * args.iterations)
    post_warmup_taus = tau_values[warmup_end:]

    # Check monotonic decrease
    for i in range(len(post_warmup_taus) - 1):
        assert post_warmup_taus[i] >= post_warmup_taus[i + 1], (
            f"Tau increased at step {i}: {post_warmup_taus[i]} -> {post_warmup_taus[i + 1]}"
        )

    # Check final tau is close to final_tau (within small tolerance due to discrete steps)
    assert abs(tau_values[-1] - args.final_tau) < 0.02, (
        f"Final tau {tau_values[-1]} doesn't match expected {args.final_tau}"
    )


def test_zero_iterations_after_warmup():
    """
    Edge case: iterations == warmup_steps should not cause division by zero.
    """
    from autoforge.Modules.Optimizer import FilamentOptimizer

    args = Namespace(
        init_tau=1.0,
        final_tau=0.01,
        iterations=10,
        warmup_fraction=1.0,  # All iterations are warmup!
        learning_rate=0.01,
        learning_rate_warmup_fraction=0.1,
        seed=42,
        tensorboard=False,
        max_layers=10,
        layer_height=0.2,
        visualize=False,
        run_name="",
        disable_visualization_for_gradio=1,
        output_folder=".",
    )

    # Create minimal setup
    H, W = 32, 32
    num_materials = 10
    material_colors = torch.rand(num_materials, 3)
    material_TDs = torch.rand(num_materials)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = torch.rand(H, W, 3).float()
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.random.randn(num_materials, 2).astype(np.float32)
    device = torch.device("cpu")

    class DummyLoss(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

    # Should not crash due to division by zero
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
        perception_loss_module=DummyLoss(),
    )

    # decay_rate should be well-defined (init_tau - final_tau) / max(1, 0) = ...
    assert not np.isnan(optimizer.decay_rate), "decay_rate is NaN"
    assert not np.isinf(optimizer.decay_rate), "decay_rate is infinite"


def test_tau_schedule_correct_calculation():
    """
    Verify the decay rate calculation is correct.
    """
    from autoforge.Modules.Optimizer import FilamentOptimizer

    args = Namespace(
        init_tau=2.0,
        final_tau=0.5,
        iterations=100,
        warmup_fraction=0.2,  # 20 steps warmup
        learning_rate=0.01,
        learning_rate_warmup_fraction=0.1,
        seed=42,
        tensorboard=False,
        max_layers=10,
        layer_height=0.2,
        visualize=False,
        run_name="",
        disable_visualization_for_gradio=1,
        output_folder=".",
    )

    # Create minimal setup
    H, W = 32, 32
    num_materials = 10
    material_colors = torch.rand(num_materials, 3)
    material_TDs = torch.rand(num_materials)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = torch.rand(H, W, 3).float()
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.random.randn(num_materials, 2).astype(np.float32)
    device = torch.device("cpu")

    class DummyLoss(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

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
        perception_loss_module=DummyLoss(),
    )

    # Expected: iterations_after_warmup = 100 - 20 = 80
    # decay_rate = (2.0 - 0.5) / 80 = 1.5 / 80 = 0.01875
    expected_decay_rate = (2.0 - 0.5) / 80
    assert abs(optimizer.decay_rate - expected_decay_rate) < 1e-6, (
        f"Decay rate {optimizer.decay_rate} doesn't match expected {expected_decay_rate}"
    )

    # At warmup end, tau should be init_tau
    optimizer.num_steps_done = 20
    tau_h, tau_g = optimizer._get_tau()
    assert tau_h == args.init_tau, (
        f"Tau at warmup end should be {args.init_tau}, got {tau_h}"
    )

    # At final step, tau should be close to final_tau (clamped)
    optimizer.num_steps_done = 99
    tau_h, tau_g = optimizer._get_tau()
    # Due to discrete steps, it might not be exactly final_tau
    assert tau_h <= args.init_tau and tau_h >= args.final_tau, (
        f"Tau at final step should be between [{args.final_tau}, {args.init_tau}], got {tau_h}"
    )


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing Bug #9: Negative Tau Schedule...")

    print("\n1. Testing negative decay rate bug (should fail before fix)...")
    try:
        test_negative_decay_rate_bug()
        print("   ❌ FAILED: Should have raised AssertionError")
    except AssertionError as e:
        if "init_tau must be >= final_tau" in str(e):
            print("   ✓ PASSED: Correctly rejected init_tau < final_tau")
        else:
            print(f"   ❌ FAILED: Wrong assertion: {e}")
    except Exception as e:
        print(f"   ⚠️  ISSUE: Unexpected error: {e}")

    print("\n2. Testing tau decreases monotonically...")
    try:
        test_tau_decreases_monotonically()
        print("   ✓ PASSED: Tau decreases correctly")
    except AssertionError as e:
        print(f"   ❌ FAILED: {e}")

    print("\n3. Testing zero iterations after warmup edge case...")
    try:
        test_zero_iterations_after_warmup()
        print("   ✓ PASSED: No division by zero")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    print("\n4. Testing tau schedule calculation...")
    try:
        test_tau_schedule_correct_calculation()
        print("   ✓ PASSED: Decay rate calculated correctly")
    except AssertionError as e:
        print(f"   ❌ FAILED: {e}")

    print("\nAll tests complete!")
