"""
Test the optimal fix: dominant dimension exact, non-dominant rounded
"""

import numpy as np


def resize_optimal(img, max_size):
    """Optimal implementation - dominant exact, non-dominant rounded"""
    h_img, w_img, _ = img.shape

    if w_img >= h_img:
        scale = max_size / w_img
        new_w = max_size
        new_h = int(round(h_img * scale))
    else:
        scale = max_size / h_img
        new_h = max_size
        new_w = int(round(w_img * scale))

    return (new_w, new_h)


def resize_old_buggy(img, max_size):
    """Old buggy - both rounded after scaling"""
    h_img, w_img, _ = img.shape

    if w_img >= h_img:
        scale = max_size / w_img
    else:
        scale = max_size / h_img

    new_w = int(round(w_img * scale))
    new_h = int(round(h_img * scale))

    return (new_w, new_h)


test_cases = [
    (100, 101, 50),
    (101, 100, 50),
    (100, 103, 50),
    (97, 100, 50),
    (91, 100, 50),
    (200, 207, 100),
]

print("Comparison: Old Buggy vs Optimal Fix")
print("=" * 80)

total_old_error = 0
total_new_error = 0

for w, h, max_size in test_cases:
    img = np.zeros((h, w, 3), dtype=np.uint8)

    old_w, old_h = resize_old_buggy(img, max_size)
    new_w, new_h = resize_optimal(img, max_size)

    orig_ratio = w / h
    old_ratio = old_w / old_h
    new_ratio = new_w / new_h

    old_error = abs(old_ratio - orig_ratio)
    new_error = abs(new_ratio - orig_ratio)

    total_old_error += old_error
    total_new_error += new_error

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
        f"  New (optimal):  {new_w}x{new_h}, ratio={new_ratio:.6f}, error={new_error:.6f}"
    )
    print(f"  {improvement} (delta={abs(new_error - old_error):.6f})")
    print()

print("=" * 80)
print(f"Total error - Old: {total_old_error:.6f}, New: {total_new_error:.6f}")
print(
    f"Average error - Old: {total_old_error / len(test_cases):.6f}, New: {total_new_error / len(test_cases):.6f}"
)

if total_new_error < total_old_error:
    print("✓ OPTIMAL FIX IS BETTER overall")
else:
    print("✗ Optimal fix is not better")
