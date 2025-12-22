#!/usr/bin/env python
"""
Test to verify Bug #10 fix is applied in OutputHelper.py
This test directly checks that the bounds check is in place.
"""

import inspect
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autoforge.Helper.OutputHelper import generate_project_file


def test_bug10_fix_code_analysis():
    """
    Directly inspect the generate_project_file source code to verify
    the bounds check is present.
    """
    print("=" * 70)
    print("BUG #10 FIX VERIFICATION: Code Analysis")
    print("=" * 70)

    # Get the source code of generate_project_file
    source = inspect.getsource(generate_project_file)

    # Check for the specific bounds check we added
    expected_checks = [
        "if not (0 <= idx < len(material_data))",
        "Invalid material index",
    ]

    print("\nChecking for bounds validation code...")
    all_found = True
    for check in expected_checks:
        if check in source:
            print(f"  ✓ Found: '{check}'")
        else:
            print(f"  ✗ Missing: '{check}'")
            all_found = False

    if all_found:
        print("\n✓ BUG #10 FIX CONFIRMED: Bounds checking code is present")

        # Show the actual code section
        print("\nVerifying context around the fix...")
        for i, line in enumerate(source.split("\n")):
            if "Invalid material index" in line:
                # Show context
                lines = source.split("\n")
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                print("\nContext:")
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"{marker} {lines[j]}")
                break

        return True
    else:
        print("\n✗ BUG #10 FIX NOT FOUND: Fix code not present")
        return False


if __name__ == "__main__":
    result = test_bug10_fix_code_analysis()
    sys.exit(0 if result else 1)
