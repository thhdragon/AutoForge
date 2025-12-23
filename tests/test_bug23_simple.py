"""Simple test to understand the exact behavior"""

import numpy as np
from autoforge.Helper.ImageHelper import resize_image

# Test 1: 100 wide x 101 tall
print("Test 1: 100x101 (height is larger)")
img1 = np.random.randint(0, 255, (101, 100, 3), dtype=np.uint8)
print(f"Original: h={101}, w={100}")
print(f"w_img >= h_img: {100 >= 101} = False, so height is dominant")
print(f"scale = 50 / 101 = {50 / 101}")
print(f"Expected: new_h = 50, new_w = int(100 * {50 / 101}) = {int(100 * 50 / 101)}")

resized1 = resize_image(img1, max_size=50)
h1, w1, _ = resized1.shape
print(f"Actual: {w1}x{h1}")
print(
    f"Aspect ratio: original={100 / 101:.6f}, new={w1 / h1:.6f}, error={abs((w1 / h1) - (100 / 101)):.6f}"
)
print()

# Test 2: 101 wide x 100 tall
print("Test 2: 101x100 (width is larger)")
img2 = np.random.randint(0, 255, (100, 101, 3), dtype=np.uint8)
print(f"Original: h={100}, w={101}")
print(f"w_img >= h_img: {101 >= 100} = True, so width is dominant")
print(f"scale = 50 / 101 = {50 / 101}")
print(f"Expected: new_w = 50, new_h = int(100 * {50 / 101}) = {int(100 * 50 / 101)}")

resized2 = resize_image(img2, max_size=50)
h2, w2, _ = resized2.shape
print(f"Actual: {w2}x{h2}")
print(
    f"Aspect ratio: original={101 / 100:.6f}, new={w2 / h2:.6f}, error={abs((w2 / h2) - (101 / 100)):.6f}"
)
print()

# Test 3: Perfect 1:1 preservation test
print("Test 3: 100x103 (height larger)")
img3 = np.random.randint(0, 255, (103, 100, 3), dtype=np.uint8)
print(f"Original: h={103}, w={100}")
print(f"w_img >= h_img: {100 >= 103} = False, so height is dominant")
print(f"scale = 50 / 103 = {50 / 103}")
print(f"Expected: new_h = 50, new_w = int(100 * {50 / 103}) = {int(100 * 50 / 103)}")

resized3 = resize_image(img3, max_size=50)
h3, w3, _ = resized3.shape
print(f"Actual: {w3}x{h3}")
print(
    f"Aspect ratio: original={100 / 103:.6f}, new={w3 / h3:.6f}, error={abs((w3 / h3) - (100 / 103)):.6f}"
)
