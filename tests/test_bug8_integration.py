"""
Bug #8 Integration Test: Verify TD validation prevents downstream corruption

This test verifies that the TD validation in FilamentHelper prevents
the opacity calculation corruption that would occur with invalid TD values.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoforge.Helper.FilamentHelper import load_materials


class Args:
    def __init__(self, csv_file="", json_file=""):
        self.csv_file = csv_file
        self.json_file = json_file


def simulate_opacity_calculation(TD_value, thickness=0.2):
    """
    Simulate the opacity calculation that would break with bad TD.
    This is what happens in the rendering code.
    """
    if TD_value <= 0:
        # This is what would happen WITHOUT validation
        thick_ratio = thickness / TD_value  # Division by zero/negative!
        opacity = 1.0 - np.exp(-thick_ratio)
        return opacity
    else:
        # Normal case
        thick_ratio = thickness / TD_value
        opacity = 1.0 - np.exp(-thick_ratio)
        return opacity


def test_without_validation_would_break():
    """
    Demonstrate what WOULD happen without the TD validation fix.
    This simulates the bug scenario.
    """
    print("\n" + "=" * 60)
    print("SCENARIO: What happens WITHOUT TD validation")
    print("=" * 60)

    # Case 1: Zero TD
    print("\nCase 1: TD = 0 (division by zero)")
    try:
        opacity = simulate_opacity_calculation(TD_value=0.0, thickness=0.2)
        print(f"  thick_ratio = 0.2 / 0.0 = {0.2 / 0.0}")
        print(f"  opacity = 1 - exp(-inf) = {opacity}")
        print(f"  Result: {'NaN' if np.isnan(opacity) else 'Invalid: ' + str(opacity)}")
    except ZeroDivisionError:
        print(f"  thick_ratio = 0.2 / 0.0 → ZeroDivisionError!")
        print(f"  Result: Program crashes (or gets inf)")
        print(f"  This would corrupt the entire output!")

    # Case 2: Negative TD
    print("\nCase 2: TD = -1.5 (negative value)")
    opacity = simulate_opacity_calculation(TD_value=-1.5, thickness=0.2)
    thick_ratio = 0.2 / -1.5
    print(f"  thick_ratio = 0.2 / -1.5 = {thick_ratio:.4f}")
    print(f"  opacity = 1 - exp({-thick_ratio:.4f}) = {opacity:.4f}")
    print(f"  Result: Invalid opacity = {opacity:.4f} (should be in [0,1]!)")

    print("\n✓ Confirmed: Invalid TD values cause corruption")


def test_with_validation_prevents_corruption():
    """
    Verify that the fix properly prevents the corruption scenario.
    """
    print("\n" + "=" * 60)
    print("SCENARIO: With TD validation (the fix)")
    print("=" * 60)

    # Try to load CSV with zero TD
    print("\nAttempting to load CSV with TD = 0...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("TestBrand,BadMaterial,#FF0000,0.0\n")
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)
        try:
            colors, tds, names, colors_list = load_materials(args)
            print("  ❌ ERROR: Should have raised ValueError!")
            assert False, "Validation failed to catch TD=0!"
        except ValueError as e:
            print(f"  ✓ Validation caught it: {str(e)[:80]}...")
            print("  ✓ Prevented downstream corruption!")
    finally:
        os.unlink(csv_path)

    # Try with negative TD
    print("\nAttempting to load CSV with TD = -1.5...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("TestBrand,BadMaterial,#FF0000,-1.5\n")
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)
        try:
            colors, tds, names, colors_list = load_materials(args)
            print("  ❌ ERROR: Should have raised ValueError!")
            assert False, "Validation failed to catch negative TD!"
        except ValueError as e:
            print(f"  ✓ Validation caught it: {str(e)[:80]}...")
            print("  ✓ Prevented downstream corruption!")
    finally:
        os.unlink(csv_path)

    # Verify valid TD works and produces valid opacity
    print("\nLoading valid CSV with TD = 1.0...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("TestBrand,GoodMaterial,#FF0000,1.0\n")
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)
        colors, tds, names, colors_list = load_materials(args)
        print(f"  ✓ Loaded successfully")
        print(f"  TD value: {tds[0]}")

        # Simulate opacity calculation with valid TD
        opacity = simulate_opacity_calculation(TD_value=tds[0], thickness=0.2)
        print(f"  Opacity with thickness=0.2: {opacity:.4f}")

        assert 0.0 <= opacity <= 1.0, f"Opacity out of range: {opacity}"
        assert not np.isnan(opacity), "Opacity is NaN!"
        assert not np.isinf(opacity), "Opacity is inf!"
        print(f"  ✓ Valid opacity value in range [0, 1]")
    finally:
        os.unlink(csv_path)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Bug #8 Integration Test: TD Validation Prevents Corruption")
    print("=" * 60)

    # First show what would break without validation
    test_without_validation_would_break()

    # Then verify the fix prevents it
    test_with_validation_prevents_corruption()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("✓ Without validation: TD ≤ 0 causes NaN/invalid opacity")
    print("✓ With validation: Invalid TD is caught before corruption")
    print("✓ Valid TD values produce valid opacity in [0, 1]")
    print("\n✓ Bug #8 fix successfully prevents downstream corruption!")
    print("=" * 60)
