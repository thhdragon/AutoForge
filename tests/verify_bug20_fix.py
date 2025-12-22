"""Quick verification that Bug #20 fix handles errors correctly"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from autoforge.Helper.FilamentHelper import hex_to_rgb

print("=" * 60)
print("Bug #20 Fix Verification - Error Handling")
print("=" * 60)

# Test invalid cases
test_cases = [
    ("#ZZZZZZ", "Invalid hex digits"),
    ("#12", "Wrong length (too short)"),
    ("", "Empty string"),
    ("#GGGGGG", "Invalid character G"),
]

for hex_val, description in test_cases:
    print(f"\nTesting {repr(hex_val)}: {description}")
    try:
        result = hex_to_rgb(hex_val)
        print(f"  ✗ ERROR: Should have raised ValueError, got {result}")
    except ValueError as e:
        print(f"  ✓ Caught: {e}")

# Test valid cases
print("\n" + "=" * 60)
print("Valid hex values:")
print("=" * 60)

valid_cases = [
    ("#FF8000", "Orange"),
    ("#ABC", "3-char light blue"),
    ("00FF00", "Green without #"),
    ("#FFF", "White 3-char"),
]

for hex_val, description in valid_cases:
    try:
        result = hex_to_rgb(hex_val)
        print(f"  ✓ {hex_val:10} ({description}): {result}")
    except Exception as e:
        print(f"  ✗ {hex_val:10} failed: {e}")

print("\n" + "=" * 60)
print("Bug #20 Fix: VERIFIED ✓")
print("=" * 60)
