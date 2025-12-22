"""
Test for Bug #8: Transmission Distance (TD) Validation

Verifies that invalid TD values (≤ 0) are properly caught and rejected
in FilamentHelper.load_materials_data_np().
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoforge.Helper.FilamentHelper import load_materials


class Args:
    """Mock args object for testing"""

    def __init__(self, csv_file="", json_file=""):
        self.csv_file = csv_file
        self.json_file = json_file


def test_valid_td_values():
    """Test that valid TD values are accepted"""
    # Create a temporary CSV with valid data
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("BrandA,Material1,#FF0000,0.5\n")
        f.write("BrandB,Material2,#00FF00,1.0\n")
        f.write("BrandC,Material3,#0000FF,2.5\n")
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)
        colors, tds, names, colors_list = load_materials(args)

        # Verify data was loaded correctly
        assert len(tds) == 3
        assert np.all(tds > 0)
        assert np.allclose(tds, [0.5, 1.0, 2.5])
        print("✓ Valid TD values accepted correctly")
    finally:
        os.unlink(csv_path)


def test_zero_td_rejected():
    """Test that TD = 0 is rejected with ValueError"""
    # Create a temporary CSV with zero TD
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("BrandA,Material1,#FF0000,0.5\n")
        f.write("BrandB,BadMaterial,#00FF00,0.0\n")  # Zero TD!
        f.write("BrandC,Material3,#0000FF,2.5\n")
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_materials(args)

        # Check error message contains useful info
        error_msg = str(exc_info.value)
        assert "Invalid Transmissivity" in error_msg
        assert "BadMaterial" in error_msg
        assert "0.0" in error_msg or "0." in error_msg
        print(f"✓ Zero TD rejected with error: {error_msg}")
    finally:
        os.unlink(csv_path)


def test_negative_td_rejected():
    """Test that negative TD values are rejected with ValueError"""
    # Create a temporary CSV with negative TD
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("BrandA,Material1,#FF0000,0.5\n")
        f.write("BrandB,NegativeMaterial,#00FF00,-1.5\n")  # Negative TD!
        f.write("BrandC,Material3,#0000FF,2.5\n")
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_materials(args)

        # Check error message contains useful info
        error_msg = str(exc_info.value)
        assert "Invalid Transmissivity" in error_msg
        assert "NegativeMaterial" in error_msg
        assert "-1.5" in error_msg
        print(f"✓ Negative TD rejected with error: {error_msg}")
    finally:
        os.unlink(csv_path)


def test_multiple_invalid_tds_rejected():
    """Test that multiple invalid TD values are all reported"""
    # Create a temporary CSV with multiple bad TDs
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("BrandA,BadMaterial1,#FF0000,0.0\n")  # Zero
        f.write("BrandB,GoodMaterial,#00FF00,1.0\n")  # Good
        f.write("BrandC,BadMaterial2,#0000FF,-2.0\n")  # Negative
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_materials(args)

        # Check error message reports both invalid materials
        error_msg = str(exc_info.value)
        assert "Invalid Transmissivity" in error_msg
        assert "BadMaterial1" in error_msg
        assert "BadMaterial2" in error_msg
        print(f"✓ Multiple invalid TDs reported: {error_msg}")
    finally:
        os.unlink(csv_path)


def test_tiny_positive_td_accepted():
    """Test that very small but positive TD values are accepted"""
    # Create a temporary CSV with tiny positive TD
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("BrandA,Material1,#FF0000,0.5\n")
        f.write("BrandB,TinyMaterial,#00FF00,0.0001\n")  # Tiny but positive
        f.write("BrandC,Material3,#0000FF,1e-6\n")  # Very tiny but positive
        csv_path = f.name

    try:
        args = Args(csv_file=csv_path)
        colors, tds, names, colors_list = load_materials(args)

        # Should succeed - tiny positive values are valid
        assert len(tds) == 3
        assert np.all(tds > 0)
        assert tds[1] == pytest.approx(0.0001)
        assert tds[2] == pytest.approx(1e-6)
        print("✓ Tiny positive TD values accepted correctly")
    finally:
        os.unlink(csv_path)


if __name__ == "__main__":
    print("Testing Bug #8 Fix: TD Validation")
    print("=" * 50)

    test_valid_td_values()
    test_zero_td_rejected()
    test_negative_td_rejected()
    test_multiple_invalid_tds_rejected()
    test_tiny_positive_td_accepted()

    print("=" * 50)
    print("All tests passed! ✓")
    print("\nBug #8 fix verified:")
    print("- Valid TD values (> 0) are accepted")
    print("- Zero TD values are rejected with clear error")
    print("- Negative TD values are rejected with clear error")
    print("- Multiple invalid TDs are all reported")
    print("- Error messages identify problematic materials")
