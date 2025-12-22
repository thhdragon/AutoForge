"""
Test for Bug #20: CSV Hex Color Parsing No Validation

This test verifies that hex_to_rgb function:
1. Currently crashes on invalid hex codes like #ZZZZZZ
2. Doesn't handle 3-char hex like #ABC
3. Needs proper validation and error messages

After the fix, it should:
- Validate hex format
- Support 3-char hex (#ABC -> #AABBCC)
- Provide clear error messages for invalid input
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from autoforge.Helper.FilamentHelper import hex_to_rgb


class TestBug20HexValidation:
    """Test suite for Bug #20: hex_to_rgb validation issues"""

    def test_invalid_hex_crashes(self):
        """Bug verification: Invalid hex like #ZZZZZZ should crash with poor error"""
        # This should fail before the fix
        with pytest.raises(ValueError) as exc_info:
            hex_to_rgb("#ZZZZZZ")

        # Before fix: generic "invalid literal for int() with base 16"
        # After fix: "Invalid hex digits in ZZZZZZ"
        error_msg = str(exc_info.value)
        print(f"Error message: {error_msg}")

        # After fix, error should mention "hex" or "Invalid"
        assert "hex" in error_msg.lower() or "invalid" in error_msg.lower()

    def test_three_char_hex_unsupported(self):
        """Bug verification: 3-char hex like #ABC not handled"""
        # Before fix: This will crash with index error
        # After fix: Should expand to #AABBCC
        try:
            result = hex_to_rgb("#ABC")
            # After fix, #ABC should expand to #AABBCC
            expected = [0xAA / 255.0, 0xBB / 255.0, 0xCC / 255.0]
            assert len(result) == 3
            for r, e in zip(result, expected):
                assert abs(r - e) < 0.001
            print(f"✓ 3-char hex #ABC correctly expanded: {result}")
        except (IndexError, ValueError) as e:
            # Before fix: Expected to fail
            print(f"✗ 3-char hex #ABC failed (bug confirmed): {e}")
            pytest.fail(f"3-char hex not supported (Bug #20): {e}")

    def test_valid_hex_works(self):
        """Valid 6-char hex should work correctly"""
        result = hex_to_rgb("#FF8000")
        expected = [1.0, 128 / 255.0, 0.0]
        assert len(result) == 3
        for r, e in zip(result, expected):
            assert abs(r - e) < 0.001

    def test_hex_without_hash(self):
        """Hex without # prefix should work"""
        result = hex_to_rgb("00FF00")
        expected = [0.0, 1.0, 0.0]
        assert len(result) == 3
        for r, e in zip(result, expected):
            assert abs(r - e) < 0.001

    def test_wrong_length_hex(self):
        """Wrong length hex should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            hex_to_rgb("#FFFF")  # Too short (4 chars)

        error_msg = str(exc_info.value)
        print(f"Error for wrong length: {error_msg}")
        # After fix, should mention "Invalid hex color"
        assert "invalid" in error_msg.lower()

    def test_empty_hex(self):
        """Empty hex string should raise ValueError"""
        with pytest.raises(ValueError):
            hex_to_rgb("")

    def test_various_invalid_chars(self):
        """Various invalid characters should raise clear errors"""
        invalid_cases = [
            "#GGGGGG",  # G is invalid
            "#12345G",  # Last char invalid
            "#!!!!!",  # Special chars
            "XYZXYZ",  # All invalid
        ]

        for invalid_hex in invalid_cases:
            with pytest.raises(ValueError) as exc_info:
                hex_to_rgb(invalid_hex)

            error_msg = str(exc_info.value)
            # After fix, should have clear error message
            assert "hex" in error_msg.lower() or "invalid" in error_msg.lower()
            print(f"✓ {invalid_hex} raised: {error_msg}")

    def test_three_char_variations(self):
        """Test various 3-char hex codes after fix"""
        test_cases = [
            ("#FFF", [1.0, 1.0, 1.0]),  # White
            ("#000", [0.0, 0.0, 0.0]),  # Black
            ("#F00", [1.0, 0.0, 0.0]),  # Red
            ("#0F0", [0.0, 1.0, 0.0]),  # Green
            ("#00F", [0.0, 0.0, 1.0]),  # Blue
            ("ABC", [0xAA / 255.0, 0xBB / 255.0, 0xCC / 255.0]),  # No hash
        ]

        for hex_str, expected in test_cases:
            try:
                result = hex_to_rgb(hex_str)
                assert len(result) == 3
                for r, e in zip(result, expected):
                    assert abs(r - e) < 0.001
                print(f"✓ {hex_str} -> {result}")
            except Exception as e:
                print(f"✗ {hex_str} failed (bug still present): {e}")
                pytest.fail(f"3-char hex {hex_str} should work after fix: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Bug #20 Verification Test")
    print("Testing hex_to_rgb validation issues")
    print("=" * 60)

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
