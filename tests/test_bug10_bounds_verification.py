"""
Test to verify Bug #10 fix: Material Index Out-of-Bounds Access

This test verifies that the bounds checking in OutputHelper.py::generate_project_file
properly validates filament_indices before accessing material_data, preventing crashes
when invalid material indices are provided.
"""

import os
import tempfile
import numpy as np
import pytest
from argparse import Namespace


def test_bug10_invalid_material_index_caught():
    """
    Test that invalid material indices are caught and raise ValueError.

    This simulates the scenario where extract_filament_swaps() returns
    indices that exceed the length of material_data.
    """
    # Import after we're in the test to ensure clean imports
    from autoforge.Helper.OutputHelper import generate_project_file

    # Create minimal test args
    args = Namespace(
        background_height=0.4,
        layer_height=0.2,
        max_layers=10,
        background_color="#FFFFFF",
        csv_file="test.csv",
        json_file="",
    )

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        project_file = os.path.join(tmpdir, "test_project.hfp")
        stl_file = os.path.join(tmpdir, "test.stl")

        # Create a simple CSV with only 2 materials (indices 0 and 1)
        csv_file = os.path.join(tmpdir, "materials.csv")
        with open(csv_file, "w") as f:
            f.write("Brand,Name,Color,Transmissivity,Type,Uuid,Owned\n")
            f.write("Test,Material0,#FF0000,0.5,PLA,uuid-0,true\n")
            f.write("Test,Material1,#00FF00,0.7,PLA,uuid-1,true\n")

        args.csv_file = csv_file

        # Create disc_global with INVALID index 5 (only have materials 0, 1)
        # This should trigger the bounds check
        disc_global = np.array([0, 1, 5, 1, 0])  # Index 5 is out of bounds!
        disc_height_image = np.ones((10, 10)) * 5  # 5 layers

        # The function should raise ValueError due to invalid index
        with pytest.raises(ValueError) as exc_info:
            generate_project_file(
                project_filename=project_file,
                args=args,
                disc_global=disc_global,
                disc_height_image=disc_height_image,
                width_mm=100.0,
                height_mm=100.0,
                stl_filename=stl_file,
                csv_filename=csv_file,
            )

        # Verify the error message is informative
        error_msg = str(exc_info.value)
        assert "Invalid material index" in error_msg
        assert "5" in error_msg  # The invalid index
        assert "2" in error_msg  # The actual number of materials

        print(f"✓ Bug #10 fix verified: Invalid index properly caught")
        print(f"  Error message: {error_msg}")


def test_bug10_valid_indices_work():
    """
    Test that valid material indices work correctly without errors.

    This is a sanity check to ensure the bounds checking doesn't
    interfere with normal operation.
    """
    from autoforge.Helper.OutputHelper import generate_project_file

    args = Namespace(
        background_height=0.4,
        layer_height=0.2,
        max_layers=10,
        background_color="#FFFFFF",
        csv_file="test.csv",
        json_file="",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_file = os.path.join(tmpdir, "test_project.hfp")
        stl_file = os.path.join(tmpdir, "test.stl")

        # Create CSV with 3 materials
        csv_file = os.path.join(tmpdir, "materials.csv")
        with open(csv_file, "w") as f:
            f.write("Brand,Name,Color,Transmissivity,Type,Uuid,Owned\n")
            f.write("Test,Red,#FF0000,0.5,PLA,uuid-0,true\n")
            f.write("Test,Green,#00FF00,0.7,PLA,uuid-1,true\n")
            f.write("Test,Blue,#0000FF,0.3,PLA,uuid-2,true\n")

        args.csv_file = csv_file

        # Valid indices only (0, 1, 2)
        disc_global = np.array([0, 1, 2, 1, 0])
        disc_height_image = np.ones((10, 10)) * 5

        # Should complete without error
        generate_project_file(
            project_filename=project_file,
            args=args,
            disc_global=disc_global,
            disc_height_image=disc_height_image,
            width_mm=100.0,
            height_mm=100.0,
            stl_filename=stl_file,
            csv_filename=csv_file,
        )

        # Verify file was created
        assert os.path.exists(project_file), "Project file should be created"

        # Load and verify it's valid JSON with expected structure
        import json

        with open(project_file, "r") as f:
            data = json.load(f)

        assert "filament_set" in data
        assert len(data["filament_set"]) > 0
        assert "slider_values" in data

        print("✓ Valid indices work correctly")
        print(f"  Generated project with {len(data['filament_set'])} filaments")


def test_bug10_edge_case_max_valid_index():
    """
    Test edge case where the maximum valid index is used.

    This ensures the bounds check uses the correct comparison (< not <=).
    """
    from autoforge.Helper.OutputHelper import generate_project_file

    args = Namespace(
        background_height=0.4,
        layer_height=0.2,
        max_layers=10,
        background_color="#FFFFFF",
        csv_file="test.csv",
        json_file="",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_file = os.path.join(tmpdir, "test_project.hfp")
        stl_file = os.path.join(tmpdir, "test.stl")

        # Create CSV with 3 materials (valid indices: 0, 1, 2)
        csv_file = os.path.join(tmpdir, "materials.csv")
        with open(csv_file, "w") as f:
            f.write("Brand,Name,Color,Transmissivity,Type,Uuid,Owned\n")
            f.write("Test,Mat0,#FF0000,0.5,PLA,uuid-0,true\n")
            f.write("Test,Mat1,#00FF00,0.7,PLA,uuid-1,true\n")
            f.write("Test,Mat2,#0000FF,0.3,PLA,uuid-2,true\n")

        args.csv_file = csv_file

        # Use maximum valid index (2, since we have 3 materials)
        disc_global = np.array([2, 2, 2])
        disc_height_image = np.ones((10, 10)) * 3

        # Should work without error
        generate_project_file(
            project_filename=project_file,
            args=args,
            disc_global=disc_global,
            disc_height_image=disc_height_image,
            width_mm=100.0,
            height_mm=100.0,
            stl_filename=stl_file,
            csv_filename=csv_file,
        )

        assert os.path.exists(project_file)
        print("✓ Maximum valid index (n-1) works correctly")


def test_bug10_negative_index_caught():
    """
    Test that negative indices are also caught by bounds checking.
    """
    from autoforge.Helper.OutputHelper import generate_project_file

    args = Namespace(
        background_height=0.4,
        layer_height=0.2,
        max_layers=10,
        background_color="#FFFFFF",
        csv_file="test.csv",
        json_file="",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        project_file = os.path.join(tmpdir, "test_project.hfp")
        stl_file = os.path.join(tmpdir, "test.stl")

        csv_file = os.path.join(tmpdir, "materials.csv")
        with open(csv_file, "w") as f:
            f.write("Brand,Name,Color,Transmissivity,Type,Uuid,Owned\n")
            f.write("Test,Material0,#FF0000,0.5,PLA,uuid-0,true\n")

        args.csv_file = csv_file

        # Negative index should be caught
        disc_global = np.array([0, -1, 0])
        disc_height_image = np.ones((10, 10)) * 3

        with pytest.raises(ValueError) as exc_info:
            generate_project_file(
                project_filename=project_file,
                args=args,
                disc_global=disc_global,
                disc_height_image=disc_height_image,
                width_mm=100.0,
                height_mm=100.0,
                stl_filename=stl_file,
                csv_filename=csv_file,
            )

        error_msg = str(exc_info.value)
        assert "Invalid material index" in error_msg
        assert "-1" in error_msg

        print("✓ Negative indices properly caught")
        print(f"  Error message: {error_msg}")


if __name__ == "__main__":
    print("=" * 70)
    print("Bug #10 Verification Tests: Material Index Out-of-Bounds Access")
    print("=" * 70)
    print()

    # Run tests
    test_bug10_invalid_material_index_caught()
    print()
    test_bug10_valid_indices_work()
    print()
    test_bug10_edge_case_max_valid_index()
    print()
    test_bug10_negative_index_caught()
    print()

    print("=" * 70)
    print("All Bug #10 tests passed! ✓")
    print("=" * 70)
