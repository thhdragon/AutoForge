"""
Test Bug 24: Silent Dominant Color Computation Failure

Verifies that exceptions in _compute_dominant_image_color are properly reported
with traceback information instead of being silently swallowed.

Bug Description:
- _compute_dominant_image_color catches all exceptions and returns None
- Caller (_auto_select_background_color) just prints a generic warning
- No traceback or actual error message is shown
- Makes debugging failures very difficult

Expected Fix:
- When res is None, capture and print the actual exception with full traceback
- Or better yet, don't catch the exception in the first place if it's unexpected
"""

import sys
import os
import numpy as np
from io import StringIO
from unittest.mock import patch, MagicMock
import traceback as tb

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


def test_bug24_exception_in_compute_dominant_color_is_silent():
    """Verify current behavior: exceptions are silently caught."""
    from autoforge.auto_forge import _compute_dominant_image_color

    # Create invalid input that will trigger an exception
    # Pass None which will cause an AttributeError when accessing .shape

    # Capture stderr to see if traceback is printed
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        result = _compute_dominant_image_color(None, None)

        # Should return None
        assert result is None, "Should return None on exception"

        # Check if traceback was printed
        stderr_content = sys.stderr.getvalue()

        # Current implementation DOES print traceback in _compute_dominant_image_color
        # But the bug is that the caller doesn't provide context
        assert "AttributeError" in stderr_content or "NoneType" in stderr_content, (
            "Traceback should be printed in _compute_dominant_image_color"
        )

        print("✓ _compute_dominant_image_color does print traceback")
        print(f"Stderr output:\n{stderr_content}")
    finally:
        sys.stderr = old_stderr


def test_bug24_caller_provides_no_context():
    """Verify the real issue: caller doesn't provide helpful context when None is returned."""
    from autoforge.auto_forge import _auto_select_background_color

    # Create mock args
    args = MagicMock()
    args.auto_background_color = True
    args.output_folder = "/tmp/test"

    # Create valid but small inputs
    img_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    alpha = None
    material_colors_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    material_names = ["Test Material"]
    colors_list = ["#808080"]

    # Patch _compute_dominant_image_color to raise an exception
    from autoforge import auto_forge

    original_func = auto_forge._compute_dominant_image_color

    def failing_compute(*args, **kwargs):
        raise ValueError("Simulated failure in color computation")

    # Capture stdout to see warning messages
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        auto_forge._compute_dominant_image_color = failing_compute

        # Call the function
        _auto_select_background_color(
            args, img_rgb, alpha, material_colors_np, material_names, colors_list
        )

        stdout_content = sys.stdout.getvalue()
        stderr_content = sys.stderr.getvalue()

        print("=== STDOUT ===", file=old_stdout)
        print(stdout_content, file=old_stdout)
        print("=== STDERR ===", file=old_stdout)
        print(stderr_content, file=old_stdout)

        # Check current behavior: generic warning but exception details are printed
        # by _compute_dominant_image_color's except block
        assert "Warning: Auto background color computation failed" in stdout_content, (
            "Should print warning message"
        )

        # The issue is that the caller doesn't know WHY it failed
        # unless you look at the traceback from _compute_dominant_image_color

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        auto_forge._compute_dominant_image_color = original_func


def test_bug24_verify_fix():
    """
    After fix: The caller should provide better context when _compute_dominant_image_color fails.

    The fix should either:
    1. Not catch exceptions in _compute_dominant_image_color (let them propagate)
    2. Or catch in caller with proper error reporting

    Currently implementing option 2: catch exceptions in caller with traceback.
    """
    from autoforge import auto_forge

    # Create mock args
    args = MagicMock()
    args.auto_background_color = True
    args.output_folder = "/tmp/test"

    # Create valid inputs
    img_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    alpha = None
    material_colors_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    material_names = ["Test Material"]
    colors_list = ["#808080"]

    # Patch to force an exception that will be caught
    original_func = auto_forge._compute_dominant_image_color

    def failing_compute(*args, **kwargs):
        raise ValueError("Test exception for verification")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        auto_forge._compute_dominant_image_color = failing_compute

        # Import after patching
        from autoforge.auto_forge import _auto_select_background_color

        # Call the function
        _auto_select_background_color(
            args, img_rgb, alpha, material_colors_np, material_names, colors_list
        )

        stdout_content = sys.stdout.getvalue()
        stderr_content = sys.stderr.getvalue()
        combined = stdout_content + stderr_content

        print("=== COMBINED OUTPUT ===", file=old_stdout)
        print(combined, file=old_stdout)

        # After fix: Should see detailed error with exception message
        # Check for the fix indicators:
        # 1. Warning message includes exception details
        # 2. Traceback is shown

        has_warning = "Warning: Auto background color" in combined
        has_exception_msg = "ValueError" in combined or "Test exception" in combined
        has_traceback = "Traceback" in combined or "File" in combined

        print(f"\nFix verification:", file=old_stdout)
        print(f"  Has warning: {has_warning}", file=old_stdout)
        print(f"  Has exception message: {has_exception_msg}", file=old_stdout)
        print(f"  Has traceback: {has_traceback}", file=old_stdout)

        # Verify the fix is working
        assert has_warning, "Should print warning"
        assert has_exception_msg, "Should include exception details"
        assert has_traceback, "Should show traceback for debugging"

        print(
            "\n✓ Fix verified: Exceptions now properly reported with context!",
            file=old_stdout,
        )

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        auto_forge._compute_dominant_image_color = original_func


def test_bug24_normal_operation_still_works():
    """Verify the fix doesn't break normal operation with valid inputs."""
    from autoforge.auto_forge import _auto_select_background_color

    # Create mock args
    args = MagicMock()
    args.auto_background_color = True
    args.output_folder = "/tmp/test_bug24"

    # Create valid inputs with actual color data
    # Create a 10x10 image that's mostly red
    img_rgb = np.full((10, 10, 3), [200, 50, 50], dtype=np.uint8)
    alpha = None
    material_colors_np = np.array(
        [
            [0.8, 0.2, 0.2],  # Reddish - should be closest
            [0.2, 0.8, 0.2],  # Greenish
            [0.2, 0.2, 0.8],  # Blueish
        ],
        dtype=np.float32,
    )
    material_names = ["Red Material", "Green Material", "Blue Material"]
    colors_list = ["#CC3333", "#33CC33", "#3333CC"]

    # Ensure output folder exists
    os.makedirs("/tmp/test_bug24", exist_ok=True)

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Call the function - should work normally
        _auto_select_background_color(
            args, img_rgb, alpha, material_colors_np, material_names, colors_list
        )

        stdout_content = sys.stdout.getvalue()

        print("=== OUTPUT ===", file=old_stdout)
        print(stdout_content, file=old_stdout)

        # Should have successful output
        assert "Auto background color: dominant image color" in stdout_content, (
            "Should successfully compute background color"
        )

        # Should have set background color on args
        assert hasattr(args, "background_color"), "Should set background_color"
        assert hasattr(args, "background_material_index"), (
            "Should set background_material_index"
        )

        # The dominant color should be close to red, so closest material should be index 0
        assert args.background_material_index == 0, (
            f"Should select red material (index 0), got {args.background_material_index}"
        )

        print("✓ Normal operation works correctly", file=old_stdout)

    finally:
        sys.stdout = old_stdout
        # Clean up
        try:
            os.remove("/tmp/test_bug24/auto_background_color.txt")
            os.rmdir("/tmp/test_bug24")
        except:
            pass


if __name__ == "__main__":
    print("Testing Bug 24: Silent Dominant Color Computation Failure\n")
    print("=" * 70)

    print("\nTest 1: Exception in _compute_dominant_image_color")
    print("-" * 70)
    test_bug24_exception_in_compute_dominant_color_is_silent()

    print("\nTest 2: Caller provides no context")
    print("-" * 70)
    test_bug24_caller_provides_no_context()

    print("\nTest 3: Verify fix")
    print("-" * 70)
    test_bug24_verify_fix()

    print("\nTest 4: Normal operation still works")
    print("-" * 70)
    test_bug24_normal_operation_still_works()

    print("\n" + "=" * 70)
    print("All tests passed!")
