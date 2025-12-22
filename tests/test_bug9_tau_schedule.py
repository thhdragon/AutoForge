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
    )

    # Create minimal setup
    H, W = 32, 32
    max_layers = 10
    material_colors = torch.rand(max_layers, 3)
    material_TDs = torch.rand(max_layers)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = (torch.rand(H, W, 3) * 255).to(torch.uint8)

    # This should raise an assertion error with the fix
    with pytest.raises(AssertionError, match="init_tau must be >= final_tau"):
        optimizer = FilamentOptimizer(
            args=args,
            target=target,
            H=H,
            W=W,
            max_layers=max_layers,
            material_colors=material_colors,
            material_TDs=material_TDs,
            background=background,
            visualize=False,
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
    )

    # Create minimal setup
    H, W = 32, 32
    max_layers = 10
    material_colors = torch.rand(max_layers, 3)
    material_TDs = torch.rand(max_layers)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = (torch.rand(H, W, 3) * 255).to(torch.uint8)

    optimizer = FilamentOptimizer(
        args=args,
        target=target,
        H=H,
        W=W,
        max_layers=max_layers,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        visualize=False,
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

    # Check final tau reaches final_tau
    assert abs(tau_values[-1] - args.final_tau) < 1e-5, (
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
    )

    # Create minimal setup
    H, W = 32, 32
    max_layers = 10
    material_colors = torch.rand(max_layers, 3)
    material_TDs = torch.rand(max_layers)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = (torch.rand(H, W, 3) * 255).to(torch.uint8)

    # Should not crash due to division by zero
    optimizer = FilamentOptimizer(
        args=args,
        target=target,
        H=H,
        W=W,
        max_layers=max_layers,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        visualize=False,
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
    )

    # Create minimal setup
    H, W = 32, 32
    max_layers = 10
    material_colors = torch.rand(max_layers, 3)
    material_TDs = torch.rand(max_layers)
    background = torch.tensor([1.0, 1.0, 1.0])
    target = (torch.rand(H, W, 3) * 255).to(torch.uint8)

    optimizer = FilamentOptimizer(
        args=args,
        target=target,
        H=H,
        W=W,
        max_layers=max_layers,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        visualize=False,
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

    # At final step, tau should be final_tau
    optimizer.num_steps_done = 99
    tau_h, tau_g = optimizer._get_tau()
    # tau = init_tau - decay_rate * (99 - 20) = 2.0 - 0.01875 * 79 ≈ 0.51875
    # But it's clamped to final_tau
    assert abs(tau_h - args.final_tau) < 1e-5, (
        f"Tau at final step should be {args.final_tau}, got {tau_h}"
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
