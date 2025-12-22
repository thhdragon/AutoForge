"""
Bug #15 Fix Proposal: Corrected Opacity Formula

The current formula has a negative linear term that causes opacity to decrease
at high thickness ratios. This is physically incorrect.

The correct formula should follow Beer-Lambert law for opacity:
    opacity = 1 - exp(-k * thickness / TD)

For our use case with thick_ratio = thickness / TD:
    opacity = 1 - exp(-k * thick_ratio)

However, we want to parameterize this to match the existing data/behavior
at small thickness values (where the current formula may have been calibrated).

PROPOSED FORMULA:
    opacity = 1 - exp(-k * thick_ratio)

where k is a calibration constant that determines how quickly opacity saturates.
- k=1: Standard Beer-Lambert (opacity=0.63 at thick_ratio=1)
- k=2: Faster saturation (opacity=0.86 at thick_ratio=1)
- k=0.5: Slower saturation (opacity=0.39 at thick_ratio=1)

Based on the current formula's behavior at small thickness, we should use k≈2.5
to give opacity≈0.92 at thick_ratio=1.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def current_opacity_formula(thick_ratio):
    """Current (buggy) opacity formula."""
    o, A, k, b = -1.2416557e-02, 9.6407950e-01, 3.4103447e01, -4.1554203e00
    opac = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    return torch.clamp(opac, 0.0, 1.0)


def proposed_opacity_formula(thick_ratio, k=2.5):
    """
    Proposed fix: Beer-Lambert based opacity.

    opacity = 1 - exp(-k * thick_ratio)

    This ensures:
    1. Monotonically increasing
    2. Starts at 0 when thick_ratio=0
    3. Asymptotically approaches 1.0
    4. Physically correct behavior

    Args:
        thick_ratio: Ratio of thickness to TD (thickness / TD)
        k: Calibration constant (default 2.5 chosen to match expected behavior)
    """
    return 1.0 - torch.exp(-k * thick_ratio)


def compare_formulas():
    """Compare current vs proposed formulas."""
    print("=" * 80)
    print("Bug #15 Fix Proposal: Opacity Formula Comparison")
    print("=" * 80)
    print()

    thick_ratios = torch.linspace(0, 3.0, 1000)
    current = current_opacity_formula(thick_ratios)

    # Test different k values
    proposed_k1 = proposed_opacity_formula(thick_ratios, k=1.0)
    proposed_k2 = proposed_opacity_formula(thick_ratios, k=2.0)
    proposed_k25 = proposed_opacity_formula(thick_ratios, k=2.5)
    proposed_k3 = proposed_opacity_formula(thick_ratios, k=3.0)

    print("COMPARISON AT KEY THICKNESS RATIOS:")
    print()
    print(
        f"{'Ratio':<8} {'Current':<12} {'k=1.0':<12} {'k=2.0':<12} {'k=2.5':<12} {'k=3.0':<12}"
    )
    print("-" * 68)

    for ratio in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        idx = int((ratio / 3.0) * 999)
        print(
            f"{ratio:<8.1f} {current[idx]:<12.4f} {proposed_k1[idx]:<12.4f} "
            f"{proposed_k2[idx]:<12.4f} {proposed_k25[idx]:<12.4f} {proposed_k3[idx]:<12.4f}"
        )

    print()
    print("ANALYSIS:")
    print()

    # Find where current formula peaks
    max_idx = torch.argmax(current).item()
    max_ratio = thick_ratios[max_idx].item()
    max_opac = current[max_idx].item()
    print(
        f"Current formula peaks at thick_ratio={max_ratio:.3f} with opacity={max_opac:.4f}"
    )
    print(f"Then DECREASES to {current[-1]:.4f} at thick_ratio=3.0")
    print()

    # Check monotonicity
    current_derivs = torch.diff(current)
    proposed_derivs = torch.diff(proposed_k25)

    current_decreasing = (current_derivs < 0).sum().item()
    proposed_decreasing = (proposed_derivs < 0).sum().item()

    print(f"Current formula has {current_decreasing} decreasing points (WRONG!)")
    print(
        f"Proposed formula (k=2.5) has {proposed_decreasing} decreasing points (correct)"
    )
    print()

    # Check gradient flow
    print("GRADIENT ANALYSIS:")
    print()
    test_ratio = torch.tensor([1.0], requires_grad=True)

    # Current formula
    curr_out = current_opacity_formula(test_ratio)
    curr_out.backward()
    curr_grad = test_ratio.grad.item()

    # Proposed formula
    test_ratio.grad = None
    prop_out = proposed_opacity_formula(test_ratio, k=2.5)
    prop_out.backward()
    prop_grad = test_ratio.grad.item()

    print(f"At thick_ratio=1.0:")
    print(f"  Current:  opacity={curr_out.item():.4f}, gradient={curr_grad:.6f}")
    print(f"  Proposed: opacity={prop_out.item():.4f}, gradient={prop_grad:.6f}")
    print()

    # Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Opacity curves
        ax = axes[0, 0]
        ax.plot(
            thick_ratios.numpy(),
            current.numpy(),
            "r-",
            linewidth=2.5,
            label="Current (Buggy)",
            alpha=0.7,
        )
        ax.plot(
            thick_ratios.numpy(),
            proposed_k1.numpy(),
            "b--",
            linewidth=1.5,
            label="Proposed k=1.0",
        )
        ax.plot(
            thick_ratios.numpy(),
            proposed_k2.numpy(),
            "g--",
            linewidth=1.5,
            label="Proposed k=2.0",
        )
        ax.plot(
            thick_ratios.numpy(),
            proposed_k25.numpy(),
            "purple",
            linewidth=2.5,
            label="Proposed k=2.5 (BEST)",
            alpha=0.8,
        )
        ax.plot(
            thick_ratios.numpy(),
            proposed_k3.numpy(),
            "orange",
            linewidth=1.5,
            label="Proposed k=3.0",
            alpha=0.6,
        )
        ax.set_xlabel("Thickness Ratio (thickness / TD)", fontsize=12)
        ax.set_ylabel("Opacity", fontsize=12)
        ax.set_title("Opacity vs Thickness Ratio", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1.05)

        # Plot 2: Derivatives
        ax = axes[0, 1]
        current_deriv = torch.diff(current).numpy()
        proposed_deriv = torch.diff(proposed_k25).numpy()
        x_deriv = thick_ratios[:-1].numpy()

        ax.plot(
            x_deriv,
            current_deriv,
            "r-",
            linewidth=2,
            label="Current (Buggy)",
            alpha=0.7,
        )
        ax.plot(
            x_deriv,
            proposed_deriv,
            "purple",
            linewidth=2,
            label="Proposed k=2.5",
            alpha=0.8,
        )
        ax.axhline(y=0, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("Thickness Ratio", fontsize=12)
        ax.set_ylabel("dOpacity / dRatio", fontsize=12)
        ax.set_title(
            "Derivative (Should Always Be Positive)", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 3)

        # Plot 3: Zoomed view at small thickness
        ax = axes[1, 0]
        mask = thick_ratios <= 0.5
        ax.plot(
            thick_ratios[mask].numpy(),
            current[mask].numpy(),
            "r-",
            linewidth=2.5,
            label="Current",
            alpha=0.7,
        )
        ax.plot(
            thick_ratios[mask].numpy(),
            proposed_k25[mask].numpy(),
            "purple",
            linewidth=2.5,
            label="Proposed k=2.5",
            alpha=0.8,
        )
        ax.set_xlabel("Thickness Ratio (thickness / TD)", fontsize=12)
        ax.set_ylabel("Opacity", fontsize=12)
        ax.set_title("Zoomed: Small Thickness Range", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 4: Error analysis
        ax = axes[1, 1]
        # At what thickness ratios would we typically operate?
        # Most prints use thickness = 0.04-0.2mm, TD = 1-6mm
        # So typical thick_ratio = 0.01 to 0.2
        typical_range = thick_ratios <= 1.0
        diff = (proposed_k25 - current).numpy()

        ax.plot(thick_ratios.numpy(), diff, "purple", linewidth=2)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.fill_between(
            thick_ratios.numpy(),
            0,
            diff,
            where=(thick_ratios.numpy() <= 1.0),
            alpha=0.3,
            label="Typical operating range",
        )
        ax.set_xlabel("Thickness Ratio", fontsize=12)
        ax.set_ylabel("Opacity Difference (Proposed - Current)", fontsize=12)
        ax.set_title("Change in Opacity from Fix", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("bug15_fix_comparison.png", dpi=150, bbox_inches="tight")
        print("✓ Plot saved to: bug15_fix_comparison.png")
        print()
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
        print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("Use the proposed formula with k=2.5:")
    print()
    print("    opacity = 1.0 - torch.exp(-2.5 * thick_ratio)")
    print()
    print("This provides:")
    print("  ✓ Monotonically increasing opacity")
    print("  ✓ Correct asymptotic behavior (approaches 1.0)")
    print("  ✓ Smooth gradients everywhere (good for optimization)")
    print("  ✓ Physically correct (Beer-Lambert law)")
    print("  ✓ Reasonable opacity values at typical thickness ratios")
    print()
    print("Alternative: If you want even faster saturation, use k=3.0")
    print("This gives opacity≈0.95 at thick_ratio=1.0")
    print()


if __name__ == "__main__":
    compare_formulas()
