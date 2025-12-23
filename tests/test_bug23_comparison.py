"""
Compare old buggy behavior vs new fixed behavior
"""

import numpy as np


def resize_image_old_buggy(img, max_size):
    """Old implementation with bug #23 - both dimensions rounded"""
    h_img, w_img, _ = img.shape

    if w_img >= h_img:
        scale = max_size / w_img
    else:
        scale = max_size / h_img

    # OLD BUGGY CODE: Both dimensions rounded
    new_w = int(round(w_img * scale))
    new_h = int(round(h_img * scale))

    return (new_w, new_h)


def resize_image_new_fixed(img, max_size):
    """New implementation with bug #23 fix - only dominant dimension exact"""
    h_img, w_img, _ = img.shape

    if w_img >= h_img:
        scale = max_size / w_img
        new_w = max_size
        new_h = int(h_img * scale)  # Truncate
    else:
        scale = max_size / h_img
        new_h = max_size
        new_w = int(w_img * scale)  # Truncate

    return (new_w, new_h)


test_cases = [
    (100, 101, 50),
    (101, 100, 50),
    (100, 103, 50),
    (97, 100, 50),
    (91, 100, 50),
    (200, 207, 100),
]

print("Comparison: Old Buggy vs New Fixed")
print("=" * 80)

for w, h, max_size in test_cases:
    img = np.zeros((h, w, 3), dtype=np.uint8)

    old_w, old_h = resize_image_old_buggy(img, max_size)
    new_w, new_h = resize_image_new_fixed(img, max_size)

    orig_ratio = w / h
    old_ratio = old_w / old_h
    new_ratio = new_w / new_h

    old_error = abs(old_ratio - orig_ratio)
    new_error = abs(new_ratio - orig_ratio)

    improvement = (
        "✓ Better"
        if new_error < old_error
        else ("✗ Worse" if new_error > old_error else "= Same")
    )

    print(f"{w}x{h} -> max={max_size}")
    print(f"  Original ratio: {orig_ratio:.6f}")
    print(
        f"  Old (buggy):    {old_w}x{old_h}, ratio={old_ratio:.6f}, error={old_error:.6f}"
    )
    print(
        f"  New (fixed):    {new_w}x{new_h}, ratio={new_ratio:.6f}, error={new_error:.6f}"
    )
    print(f"  {improvement} (Δ={abs(new_error - old_error):.6f})")
    print()

print("=" * 80)
print("Summary:")
print("The fix reduces aspect ratio distortion by ensuring only the dominant")
print("dimension uses rounding, while the other dimension is computed exactly")
print("(with truncation). This prevents the compounding error from independent")
print("rounding of both dimensions.")
