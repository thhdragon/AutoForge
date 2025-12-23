"""
Trace through original buggy code logic
"""

# Test case: 100 wide x 101 tall, max_size=50
w_img = 100
h_img = 101
max_size = 50

print("Test: 100x101 (WxH), max_size=50")
print()

# Original buggy code:
if w_img >= h_img:
    scale = max_size / w_img
    print(f"w_img >= h_img: {w_img} >= {h_img} = False")
else:
    scale = max_size / h_img
    print(f"h_img > w_img: {h_img} > {w_img} = True")

print(f"scale = {max_size} / {h_img} = {scale}")

# Buggy: both rounded
new_w_buggy = int(round(w_img * scale))
new_h_buggy = int(round(h_img * scale))

print(
    f"Buggy: new_w = round({w_img} * {scale}) = round({w_img * scale}) = {new_w_buggy}"
)
print(
    f"Buggy: new_h = round({h_img} * {scale}) = round({h_img * scale}) = {new_h_buggy}"
)
print()

# Note: h_img * scale = 101 * (50/101) = 50.0 exactly!
print(f"Note: {h_img} * {scale} = {h_img * scale} (exactly {max_size}!)")
print()

# So in this case, the dominant dimension's scaling gives an exact integer
# This masks the bug!

# Try a case where it doesn't:
print("=" * 60)
print("Test: 100x103 (WxH), max_size=50")
print()

w_img = 100
h_img = 103
max_size = 50

if w_img >= h_img:
    scale = max_size / w_img
else:
    scale = max_size / h_img

print(f"scale = {max_size} / {h_img} = {scale}")

new_w_buggy = int(round(w_img * scale))
new_h_buggy = int(round(h_img * scale))

print(
    f"Buggy: new_w = round({w_img} * {scale}) = round({w_img * scale}) = {new_w_buggy}"
)
print(
    f"Buggy: new_h = round({h_img} * {scale}) = round({h_img * scale}) = {new_h_buggy}"
)

# With optimal fix:
new_h_optimal = max_size
new_w_optimal = int(round(w_img * scale))

print(f"Optimal: new_h = {max_size} (exact)")
print(
    f"Optimal: new_w = round({w_img} * {scale}) = round({w_img * scale}) = {new_w_optimal}"
)
print()

print(f"Result: Both give {new_w_optimal}x{new_h_optimal}")
