#!/usr/bin/env python3
"""
Test script to verify Bug #8 fix: Transmission Distance (TD) Validation

Bug #8 Issue: No check for TD ≤ 0 in FilamentHelper.load_materials()
- If TD ≤ 0 → thickness / TD = inf or -inf
- log1p(34.1 * inf) = inf → opac = NaN
- Output becomes corrupted

This test verifies that invalid Transmissivity values are now caught.
"""

import sys
import tempfile
import types
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas
from autoforge.Helper.FilamentHelper import load_materials


def test_valid_transmissivity():
    """Test that valid Transmissivity values are accepted"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "materials.csv"
        df = pandas.DataFrame(
            [
                {
                    "Brand": "BrandA",
                    "Name": "Mat1",
                    "Transmissivity": 0.5,
                    "Color": "#112233",
                },
                {
                    "Brand": "BrandB",
                    "Name": "Mat2",
                    "Transmissivity": 1.0,
                    "Color": "#445566",
                },
                {
                    "Brand": "BrandC",
                    "Name": "Mat3",
                    "Transmissivity": 0.1,
                    "Color": "#778899",
                },
            ]
        )
        df.to_csv(csv_path, index=False)
        args = types.SimpleNamespace(csv_file=str(csv_path), json_file="")

        try:
            colors, tds, names, colors_list = load_materials(args)
            assert len(names) == 3
            assert len(tds) == 3
            assert all(td > 0 for td in tds)
            print("✓ Test 1 PASSED: Valid Transmissivity values accepted")
            print(f"  Loaded {len(names)} materials with TDs: {list(tds)}")
            return True
        except Exception as e:
            print(f"✗ Test 1 FAILED: {e}")
            return False


def test_zero_transmissivity():
    """Test that zero Transmissivity raises ValueError"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "materials.csv"
        df = pandas.DataFrame(
            [
                {
                    "Brand": "BrandA",
                    "Name": "Mat1",
                    "Transmissivity": 0.5,
                    "Color": "#112233",
                },
                {
                    "Brand": "BrandB",
                    "Name": "BadMat",
                    "Transmissivity": 0.0,
                    "Color": "#445566",
                },
            ]
        )
        df.to_csv(csv_path, index=False)
        args = types.SimpleNamespace(csv_file=str(csv_path), json_file="")

        try:
            colors, tds, names, colors_list = load_materials(args)
            print(
                "✗ Test 2 FAILED: Should have raised ValueError for zero Transmissivity"
            )
            return False
        except ValueError as e:
            error_msg = str(e)
            # Verify error message contains useful info
            if "Invalid Transmissivity" in error_msg and "BadMat" in error_msg:
                print("✓ Test 2 PASSED: ValueError raised for zero Transmissivity")
                print(f"  Error caught invalid material: BadMat with value 0.0")
                return True
            else:
                print(f"✗ Test 2 FAILED: Error message unclear: {error_msg[:100]}")
                return False
        except Exception as e:
            print(f"✗ Test 2 FAILED with unexpected error: {e}")
            return False


def test_negative_transmissivity():
    """Test that negative Transmissivity raises ValueError"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "materials.csv"
        df = pandas.DataFrame(
            [
                {
                    "Brand": "BrandA",
                    "Name": "Mat1",
                    "Transmissivity": 0.5,
                    "Color": "#112233",
                },
                {
                    "Brand": "BrandB",
                    "Name": "NegMat",
                    "Transmissivity": -0.5,
                    "Color": "#445566",
                },
            ]
        )
        df.to_csv(csv_path, index=False)
        args = types.SimpleNamespace(csv_file=str(csv_path), json_file="")

        try:
            colors, tds, names, colors_list = load_materials(args)
            print(
                "✗ Test 3 FAILED: Should have raised ValueError for negative Transmissivity"
            )
            return False
        except ValueError as e:
            error_msg = str(e)
            # Verify error message contains useful info
            if "Invalid Transmissivity" in error_msg and "NegMat" in error_msg:
                print("✓ Test 3 PASSED: ValueError raised for negative Transmissivity")
                print(f"  Error caught invalid material: NegMat with value -0.5")
                return True
            else:
                print(f"✗ Test 3 FAILED: Error message unclear: {error_msg[:100]}")
                return False
        except Exception as e:
            print(f"✗ Test 3 FAILED with unexpected error: {e}")
            return False


def test_multiple_invalid_values():
    """Test that multiple invalid values are all reported"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "materials.csv"
        df = pandas.DataFrame(
            [
                {
                    "Brand": "BrandA",
                    "Name": "Mat1",
                    "Transmissivity": 0.5,
                    "Color": "#112233",
                },
                {
                    "Brand": "BrandB",
                    "Name": "Bad1",
                    "Transmissivity": 0.0,
                    "Color": "#445566",
                },
                {
                    "Brand": "BrandC",
                    "Name": "Bad2",
                    "Transmissivity": -1.0,
                    "Color": "#778899",
                },
            ]
        )
        df.to_csv(csv_path, index=False)
        args = types.SimpleNamespace(csv_file=str(csv_path), json_file="")

        try:
            colors, tds, names, colors_list = load_materials(args)
            print(
                "✗ Test 4 FAILED: Should have raised ValueError for multiple invalid values"
            )
            return False
        except ValueError as e:
            error_msg = str(e)
            # Verify error message reports all invalid materials
            if "Bad1" in error_msg and "Bad2" in error_msg:
                print("✓ Test 4 PASSED: ValueError reports all invalid materials")
                print(f"  Error caught: Bad1 and Bad2")
                return True
            else:
                print(f"✗ Test 4 FAILED: Not all invalid materials reported")
                return False
        except Exception as e:
            print(f"✗ Test 4 FAILED with unexpected error: {e}")
            return False


if __name__ == "__main__":
    print("=" * 70)
    print("Bug #8 Fix Verification: Transmission Distance (TD) Validation")
    print("=" * 70)
    print()

    results = []
    results.append(test_valid_transmissivity())
    print()
    results.append(test_zero_transmissivity())
    print()
    results.append(test_negative_transmissivity())
    print()
    results.append(test_multiple_invalid_values())

    print()
    print("=" * 70)
    if all(results):
        print(f"✓ ALL TESTS PASSED ({sum(results)}/{len(results)})")
        print("Bug #8 has been successfully fixed!")
        sys.exit(0)
    else:
        print(f"✗ SOME TESTS FAILED ({sum(results)}/{len(results)})")
        sys.exit(1)
