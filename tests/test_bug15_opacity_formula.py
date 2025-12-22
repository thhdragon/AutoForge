"""
Test for Bug #15: Opacity Formula Asymptotic Behavior

The current formula is:
    opac = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    where o=-1.24e-2, A=9.64e-1, k=3.41e1, b=-4.16

The issue: The linear term `b * thick_ratio` with b=-4.16 makes opacity
decrease at large thickness values (physically incorrect).

TD (Transmissivity Distance) is the thickness at which opacity reaches
a certain level. The formula should:
1. Start near 0 when thick_ratio is 0
2. Increase monotonically as thick_ratio increases
3. Asymptotically approach 1.0 for large thick_ratio
4. Never decrease
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def current_opacity_formula(thick_ratio):
    """Current (potentially buggy) opacity formula."""
    o, A, k, b = -1.2416557e-02, 9.6407950e-01, 3.4103447e01, -4.1554203e00
    opac = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    return torch.clamp(opac, 0.0, 1.0)


def proposed_opacity_formula_v1(thick_ratio):
    """
    Proposed fix v1: Remove the linear term, use only logarithmic growth.
    This ensures monotonic increase and saturation.
    """
    o, A, k = -1.2416557e-02, 9.6407950e-01, 3.4103447e01
    opac = o + A * torch.log1p(k * thick_ratio)
    return torch.clamp(opac, 0.0, 1.0)


def proposed_opacity_formula_v2(thick_ratio):
    """
    Proposed fix v2: Use a saturating function that better models opacity.
    Opacity = 1 - exp(-thick_ratio / TD_normalized)
    This is the Beer-Lambert law for opacity.
    """
    # Normalize so that thick_ratio=1 gives opacity around 0.63 (1-1/e)
    opac = 1.0 - torch.exp(-thick_ratio)
    return opac


def proposed_opacity_formula_v3(thick_ratio):
    """
    Proposed fix v3: Use tanh for smooth saturation.
    This provides a smooth S-curve that saturates at 1.0.
    """
    # Scale to make thick_ratio=1 give opacity around 0.76
    opac = torch.tanh(1.5 * thick_ratio)
    return opac


def analyze_formulas():
    """Analyze the behavior of all opacity formulas."""
    print("=" * 80)
    print("Bug #15: Opacity Formula Analysis")
    print("=" * 80)
    print()

    # Test range: from 0 to 3.0 (thick_ratio = thickness / TD)
    # thick_ratio = 1.0 means thickness equals TD
    thick_ratios = torch.linspace(0, 3.0, 100)

    current_opac = current_opacity_formula(thick_ratios)
    proposed_v1 = proposed_opacity_formula_v1(thick_ratios)
    proposed_v2 = proposed_opacity_formula_v2(thick_ratios)
    proposed_v3 = proposed_opacity_formula_v3(thick_ratios)

    # Find where current formula starts decreasing
    derivatives = torch.diff(current_opac)
    decreasing_indices = torch.where(derivatives < 0)[0]

    print("CURRENT FORMULA ANALYSIS:")
    print(f"  Formula: o + (A * log1p(k * ratio) + b * ratio)")
    print(f"  Coefficients: o={-1.2416557e-02:.6e}, A={9.6407950e-01:.6e},")
    print(f"                k={3.4103447e01:.6e}, b={-4.1554203e00:.6e}")
    print()

    print("Key Values:")
    for ratio in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]:
        idx = int((ratio / 3.0) * 99)
        print(f"  thick_ratio={ratio:.1f}: opacity={current_opac[idx]:.4f}")
    print()

    if len(decreasing_indices) > 0:
        first_decrease_idx = decreasing_indices[0].item()
        ratio_at_decrease = thick_ratios[first_decrease_idx].item()
        print(f"⚠️  PROBLEM DETECTED:")
        print(f"  Opacity starts DECREASING at thick_ratio={ratio_at_decrease:.4f}")
        print(f"  This is physically incorrect!")
        print()

    # Check maximum opacity
    max_opac = torch.max(current_opac).item()
    max_idx = torch.argmax(current_opac).item()
    max_ratio = thick_ratios[max_idx].item()
    print(f"  Maximum opacity: {max_opac:.4f} at thick_ratio={max_ratio:.4f}")
    print(f"  Final opacity (thick_ratio=3.0): {current_opac[-1]:.4f}")
    print()

    # Analyze derivative at key points
    print("Derivative Analysis (should be positive everywhere):")
    for ratio in [0.1, 0.5, 1.0, 2.0]:
        idx = int((ratio / 3.0) * 99)
        if idx > 0:
            deriv = (current_opac[idx] - current_opac[idx - 1]).item()
            status = "✓ OK" if deriv > 0 else "✗ PROBLEM"
            print(f"  At thick_ratio={ratio:.1f}: derivative={deriv:+.6f} {status}")
    print()

    print("=" * 80)
    print("PROPOSED FORMULAS:")
    print("=" * 80)
    print()

    print("V1: Remove linear term (log only)")
    for ratio in [0.0, 0.5, 1.0, 2.0, 3.0]:
        idx = int((ratio / 3.0) * 99)
        print(f"  thick_ratio={ratio:.1f}: opacity={proposed_v1[idx]:.4f}")
    print()

    print("V2: Beer-Lambert (1 - exp(-ratio))")
    for ratio in [0.0, 0.5, 1.0, 2.0, 3.0]:
        idx = int((ratio / 3.0) * 99)
        print(f"  thick_ratio={ratio:.1f}: opacity={proposed_v2[idx]:.4f}")
    print()

    print("V3: Tanh (smooth saturation)")
    for ratio in [0.0, 0.5, 1.0, 2.0, 3.0]:
        idx = int((ratio / 3.0) * 99)
        print(f"  thick_ratio={ratio:.1f}: opacity={proposed_v3[idx]:.4f}")
    print()

    # Plot comparison
    try:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(
            thick_ratios.numpy(),
            current_opac.numpy(),
            "r-",
            linewidth=2,
            label="Current (Buggy)",
        )
        plt.plot(
            thick_ratios.numpy(),
            proposed_v1.numpy(),
            "b--",
            linewidth=2,
            label="Proposed V1 (Log only)",
        )
        plt.plot(
            thick_ratios.numpy(),
            proposed_v2.numpy(),
            "g--",
            linewidth=2,
            label="Proposed V2 (Beer-Lambert)",
        )
        plt.plot(
            thick_ratios.numpy(),
            proposed_v3.numpy(),
            "m--",
            linewidth=2,
            label="Proposed V3 (Tanh)",
        )
        plt.xlabel("Thickness Ratio (thickness / TD)")
        plt.ylabel("Opacity")
        plt.title("Opacity vs Thickness Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        # Plot derivatives
        current_deriv = torch.diff(current_opac)
        v1_deriv = torch.diff(proposed_v1)
        v2_deriv = torch.diff(proposed_v2)
        v3_deriv = torch.diff(proposed_v3)

        x_deriv = thick_ratios[:-1].numpy()
        plt.plot(
            x_deriv, current_deriv.numpy(), "r-", linewidth=2, label="Current (Buggy)"
        )
        plt.plot(x_deriv, v1_deriv.numpy(), "b--", linewidth=2, label="Proposed V1")
        plt.plot(x_deriv, v2_deriv.numpy(), "g--", linewidth=2, label="Proposed V2")
        plt.plot(x_deriv, v3_deriv.numpy(), "m--", linewidth=2, label="Proposed V3")
        plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
        plt.xlabel("Thickness Ratio (thickness / TD)")
        plt.ylabel("Derivative (dOpacity/dRatio)")
        plt.title("Derivative Analysis (should be positive)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("tests/bug15_opacity_analysis.png", dpi=150, bbox_inches="tight")
        print("✓ Plot saved to: tests/bug15_opacity_analysis.png")
        print()
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
        print()


def test_physical_properties():
    """Test that the opacity formula has correct physical properties."""
    print("=" * 80)
    print("PHYSICAL PROPERTY TESTS")
    print("=" * 80)
    print()

    tests_passed = 0
    tests_failed = 0

    # Test 1: Opacity at zero thickness should be near zero
    print("Test 1: Opacity at zero thickness")
    opac_0 = current_opacity_formula(torch.tensor([0.0])).item()
    if abs(opac_0) < 0.1:
        print(f"  ✓ PASS: opacity(0) = {opac_0:.6f} ≈ 0")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: opacity(0) = {opac_0:.6f} (expected ≈ 0)")
        tests_failed += 1
    print()

    # Test 2: Opacity should be monotonically increasing
    print("Test 2: Monotonicity (opacity should always increase)")
    ratios = torch.linspace(0, 3.0, 1000)
    opacities = current_opacity_formula(ratios)
    derivatives = torch.diff(opacities)
    negative_derivs = (derivatives < -1e-6).sum().item()

    if negative_derivs == 0:
        print(f"  ✓ PASS: Opacity is monotonically increasing")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: Found {negative_derivs} points where opacity decreases")
        tests_failed += 1
    print()

    # Test 3: Opacity should saturate (approach a limit)
    print("Test 3: Saturation (opacity should approach limit)")
    opac_large = current_opacity_formula(torch.tensor([10.0])).item()
    opac_very_large = current_opacity_formula(torch.tensor([100.0])).item()
    change = abs(opac_very_large - opac_large)

    if change < 0.01:
        print(f"  ✓ PASS: Opacity saturates (change from 10→100: {change:.6f})")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: Opacity doesn't saturate (change from 10→100: {change:.6f})")
        tests_failed += 1
    print()

    # Test 4: Opacity at thick_ratio=1 should be reasonable (0.5-0.9)
    print("Test 4: Opacity at thick_ratio=1 (thickness = TD)")
    opac_1 = current_opacity_formula(torch.tensor([1.0])).item()
    if 0.5 <= opac_1 <= 0.9:
        print(f"  ✓ PASS: opacity(1) = {opac_1:.4f} is reasonable")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: opacity(1) = {opac_1:.4f} (expected 0.5-0.9)")
        tests_failed += 1
    print()

    print("=" * 80)
    print(f"SUMMARY: {tests_passed} passed, {tests_failed} failed")
    print("=" * 80)
    print()

    return tests_failed == 0


if __name__ == "__main__":
    analyze_formulas()
    all_passed = test_physical_properties()

    if not all_passed:
        print("\n⚠️  BUG CONFIRMED: The current opacity formula has incorrect behavior!")
        print(
            "    The linear term with negative coefficient causes opacity to decrease"
        )
        print("    at large thickness values, which is physically incorrect.")
    else:
        print("\n✓ All tests passed!")
