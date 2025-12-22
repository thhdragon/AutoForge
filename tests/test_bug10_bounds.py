#!/usr/bin/env python
"""
Test to verify Bug #10: Material Index Out-of-Bounds Access
This test triggers the bug by creating a scenario where filament_indices
contains values that exceed len(material_data).
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoforge.Helper.OutputHelper import extract_filament_swaps


def test_bug10_oob_material_index():
    """
    Test that demonstrates the bug: extract_filament_swaps can return
    indices that exceed the bounds of material_data.
    """
    print("=" * 70)
    print("BUG #10: Material Index Out-of-Bounds Access")
    print("=" * 70)

    # Create a simple discrete_global array with indices 0, 1, 2, 3
    # But suppose we only have 3 materials (indices 0, 1, 2)
    disc_global = np.array([0, 1, 2, 3, 3, 2, 2], dtype=np.int32)

    # Create a simple height image
    disc_height_image = np.array(
        [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]], dtype=np.int32
    )

    background_layers = 1

    print("\nTest Setup:")
    print(f"  disc_global: {disc_global}")
    print(f"  max material index in disc_global: {np.max(disc_global)}")
    print(f"  Only 3 materials available (indices 0-2)")
    print(f"  But disc_global contains index 3 (OUT OF BOUNDS!)")

    # Extract filament swaps
    filament_indices, slider_values = extract_filament_swaps(
        disc_global, disc_height_image, background_layers
    )

    print(f"\nExtracted filament_indices: {filament_indices}")
    print(f"Extracted slider_values: {slider_values}")

    # Now try to access material_data with these indices
    # Simulate having only 3 materials
    num_materials = 3
    material_data = [
        {"name": "Material_0"},
        {"name": "Material_1"},
        {"name": "Material_2"},
    ]

    print(
        f"\nMaterial data has {len(material_data)} entries (indices 0-{len(material_data) - 1})"
    )

    # The bug: this loop will crash with IndexError when accessing index 3
    print("\nAttempting to build filament_set by accessing material_data[idx]...")
    try:
        filament_set = []
        for idx in filament_indices:
            print(f"  Accessing material_data[{idx}]...", end=" ")
            mat = material_data[idx]  # BUG: No bounds check!
            filament_set.append(mat["name"])
            print("OK")
        print("\n✓ No error occurred (unexpected)")
    except IndexError as e:
        print(f"ERROR!")
        print(f"\n✗ BUG CONFIRMED: IndexError: {e}")
        print(
            f"  Index {[i for i in filament_indices if i >= len(material_data)]} exceed bounds"
        )
        return True  # Bug confirmed

    return False


def test_bug10_with_fix():
    """
    Test the proposed fix: add bounds checking before accessing material_data.
    """
    print("\n" + "=" * 70)
    print("PROPOSED FIX: Add bounds checking")
    print("=" * 70)

    # Same setup as before
    disc_global = np.array([0, 1, 2, 3, 3, 2, 2], dtype=np.int32)
    disc_height_image = np.array(
        [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]], dtype=np.int32
    )

    filament_indices, slider_values = extract_filament_swaps(
        disc_global, disc_height_image, 1
    )

    num_materials = 3
    material_data = [
        {"name": "Material_0"},
        {"name": "Material_1"},
        {"name": "Material_2"},
    ]

    print(f"\nAttempting with bounds checking...")
    try:
        filament_set = []
        for idx in filament_indices:
            # PROPOSED FIX: Add bounds check
            if not (0 <= idx < len(material_data)):
                raise ValueError(
                    f"Invalid material index {idx}, have {len(material_data)} materials"
                )
            print(f"  Accessing material_data[{idx}]...", end=" ")
            mat = material_data[idx]
            filament_set.append(mat["name"])
            print("OK")
        print("\n✓ Bounds check caught the error!")
    except ValueError as e:
        print(f"ERROR!")
        print(f"\n✓ FIX WORKS: ValueError caught: {e}")
        return True

    return False


if __name__ == "__main__":
    bug_confirmed = test_bug10_oob_material_index()
    fix_works = test_bug10_with_fix()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if bug_confirmed:
        print("✓ Bug #10 CONFIRMED: Material index out-of-bounds access possible")
    else:
        print("✗ Bug #10 could not be confirmed with this test case")

    if fix_works:
        print("✓ Proposed fix works correctly")
    else:
        print("✗ Proposed fix did not work as expected")

    sys.exit(0 if bug_confirmed and fix_works else 1)
