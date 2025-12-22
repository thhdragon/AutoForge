#!/usr/bin/env python
"""
Test to verify Bug #10 fix is applied in OutputHelper.py
This test imports the actual generate_project_file function and verifies
it handles out-of-bounds material indices correctly.
"""

import json
import tempfile
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoforge.Helper.OutputHelper import generate_project_file


def test_bug10_fix_in_generate_project_file():
    """
    Test that generate_project_file properly validates material indices.
    """
    print("=" * 70)
    print("TESTING BUG #10 FIX IN generate_project_file()")
    print("=" * 70)

    # Create mock args
    args = Mock()
    args.background_height = 0.2
    args.layer_height = 0.1
    args.max_layers = 100
    args.background_color = "#ffffff"
    args.background_material_index = None

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hfp", delete=False) as f:
        project_filename = f.name

    # Create discrete global array with out-of-bounds index
    disc_global = np.array([0, 1, 2, 3, 3, 2, 2], dtype=np.int32)

    # Create height image
    disc_height_image = np.array(
        [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]], dtype=np.int32
    )

    # Create mock material data with only 3 materials (0-2)
    # But disc_global contains index 3
    def mock_load_materials_data(args_param):
        return [
            {
                "Brand": "TestBrand",
                "Color": "#ff0000",
                "Name": "Red",
                "Owned": False,
                "Transmissivity": 1.0,
                "Type": "PLA",
                "Uuid": "uuid-0",
            },
            {
                "Brand": "TestBrand",
                "Color": "#00ff00",
                "Name": "Green",
                "Owned": False,
                "Transmissivity": 1.0,
                "Type": "PLA",
                "Uuid": "uuid-1",
            },
            {
                "Brand": "TestBrand",
                "Color": "#0000ff",
                "Name": "Blue",
                "Owned": False,
                "Transmissivity": 1.0,
                "Type": "PLA",
                "Uuid": "uuid-2",
            },
        ]

    # Patch load_materials_data
    import autoforge.Helper.OutputHelper as output_helper

    original_load = output_helper.load_materials_data
    output_helper.load_materials_data = mock_load_materials_data

    try:
        print(
            "\nTest Case: disc_global contains index 3, but only 3 materials (0-2) available"
        )
        print(f"  disc_global: {disc_global}")
        print(f"  Number of materials: 3")

        # This should raise ValueError with proper bounds checking
        print("\nCalling generate_project_file()...")
        try:
            generate_project_file(
                project_filename,
                args,
                disc_global,
                disc_height_image,
                100.0,  # width_mm
                100.0,  # height_mm
                "dummy.stl",
                "dummy.csv",
            )
            print("✗ FAILED: No error raised (fix not applied)")
            return False
        except ValueError as e:
            error_msg = str(e)
            if "Invalid material index" in error_msg and "3" in error_msg:
                print(f"✓ PASSED: ValueError raised with correct message")
                print(f"  Error message: {error_msg}")
                return True
            else:
                print(f"✗ FAILED: Wrong error message: {error_msg}")
                return False
        except Exception as e:
            print(f"✗ FAILED: Unexpected exception: {type(e).__name__}: {e}")
            return False
    finally:
        # Restore original function
        output_helper.load_materials_data = original_load

        # Clean up temp file
        try:
            Path(project_filename).unlink()
        except:
            pass


def test_bug10_fix_with_valid_indices():
    """
    Test that generate_project_file still works correctly with valid indices.
    """
    print("\n" + "=" * 70)
    print("TESTING BUG #10 FIX WITH VALID INDICES")
    print("=" * 70)

    # Create mock args
    args = Mock()
    args.background_height = 0.2
    args.layer_height = 0.1
    args.max_layers = 100
    args.background_color = "#ffffff"
    args.background_material_index = None

    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hfp", delete=False) as f:
        project_filename = f.name

    # Create discrete global array with VALID indices (0-2)
    disc_global = np.array([0, 1, 2, 2, 1, 0], dtype=np.int32)

    # Create height image
    disc_height_image = np.array(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], dtype=np.int32
    )

    def mock_load_materials_data(args_param):
        return [
            {
                "Brand": "TestBrand",
                "Color": "#ff0000",
                "Name": "Red",
                "Owned": False,
                "Transmissivity": "1.0",
                "Type": "PLA",
                "Uuid": "uuid-0",
            },
            {
                "Brand": "TestBrand",
                "Color": "#00ff00",
                "Name": "Green",
                "Owned": False,
                "Transmissivity": "1.0",
                "Type": "PLA",
                "Uuid": "uuid-1",
            },
            {
                "Brand": "TestBrand",
                "Color": "#0000ff",
                "Name": "Blue",
                "Owned": False,
                "Transmissivity": "1.0",
                "Type": "PLA",
                "Uuid": "uuid-2",
            },
        ]

    # Patch load_materials_data
    import autoforge.Helper.OutputHelper as output_helper

    original_load = output_helper.load_materials_data
    output_helper.load_materials_data = mock_load_materials_data

    try:
        print(
            "\nTest Case: disc_global has valid indices (0-2) and 3 materials available"
        )
        print(f"  disc_global: {disc_global}")
        print(f"  Number of materials: 3")

        print("\nCalling generate_project_file()...")
        try:
            generate_project_file(
                project_filename,
                args,
                disc_global,
                disc_height_image,
                100.0,  # width_mm
                100.0,  # height_mm
                "dummy.stl",
                "dummy.csv",
            )

            # Verify the file was created and contains valid JSON
            with open(project_filename, "r") as f:
                data = json.load(f)

            if "filament_set" in data:
                print(f"✓ PASSED: Project file generated successfully")
                print(f"  Filament set entries: {len(data['filament_set'])}")
                return True
            else:
                print(f"✗ FAILED: Project file missing 'filament_set' key")
                return False
        except Exception as e:
            print(f"✗ FAILED: Exception raised: {type(e).__name__}: {e}")
            return False
    finally:
        # Restore original function
        output_helper.load_materials_data = original_load

        # Clean up temp file
        try:
            Path(project_filename).unlink()
        except:
            pass


if __name__ == "__main__":
    test1_passed = test_bug10_fix_in_generate_project_file()
    test2_passed = test_bug10_fix_with_valid_indices()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if test1_passed:
        print("✓ Test 1 PASSED: Fix detects out-of-bounds indices")
    else:
        print("✗ Test 1 FAILED")

    if test2_passed:
        print("✓ Test 2 PASSED: Fix doesn't break normal operation")
    else:
        print("✗ Test 2 FAILED")

    if test1_passed and test2_passed:
        print("\n✓ BUG #10 FIX VERIFIED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\n✗ BUG #10 FIX VERIFICATION FAILED")
        sys.exit(1)
