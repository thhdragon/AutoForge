"""
Integration test - verify resize_image works correctly with real image operations
"""

import numpy as np
from autoforge.Helper.ImageHelper import resize_image

print("Integration Test: resize_image() function")
print("=" * 60)

# Test 1: Standard usage
print("\n1. Standard resize test...")
img1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
result1 = resize_image(img1, max_size=512)
h1, w1, c1 = result1.shape
print(f"   Input: 1920x1080, Output: {w1}x{h1}")
assert w1 == 512, f"Expected width 512, got {w1}"
assert c1 == 3, f"Expected 3 channels, got {c1}"
print("   PASS: Width correctly set to max_size=512")

# Test 2: Height-dominant image
print("\n2. Height-dominant image test...")
img2 = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
result2 = resize_image(img2, max_size=512)
h2, w2, c2 = result2.shape
print(f"   Input: 1080x1920, Output: {w2}x{h2}")
assert h2 == 512, f"Expected height 512, got {h2}"
assert c2 == 3, f"Expected 3 channels, got {c2}"
print("   PASS: Height correctly set to max_size=512")

# Test 3: Square image
print("\n3. Square image test...")
img3 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
result3 = resize_image(img3, max_size=128)
h3, w3, c3 = result3.shape
print(f"   Input: 256x256, Output: {w3}x{h3}")
assert w3 == 128 and h3 == 128, f"Expected 128x128, got {w3}x{h3}"
print("   PASS: Square image remains square")

# Test 4: Aspect ratio preservation
print("\n4. Aspect ratio preservation test...")
img4 = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
result4 = resize_image(img4, max_size=400)
h4, w4, c4 = result4.shape
print(f"   Input: 1600x900 (16:9), Output: {w4}x{h4}")
expected_ratio = 1600 / 900
actual_ratio = w4 / h4
error = abs(actual_ratio - expected_ratio)
print(f"   Expected ratio: {expected_ratio:.4f}, Actual: {actual_ratio:.4f}")
print(f"   Error: {error:.4f} ({error * 100:.2f}%)")
assert error < 0.02, f"Aspect ratio error too large: {error}"
print("   PASS: Aspect ratio preserved within acceptable range")

# Test 5: Small image
print("\n5. Small image test...")
img5 = np.random.randint(0, 255, (10, 15, 3), dtype=np.uint8)
result5 = resize_image(img5, max_size=100)
h5, w5, c5 = result5.shape
print(f"   Input: 15x10, Output: {w5}x{h5}")
assert w5 == 100, f"Expected width 100, got {w5}"
print("   PASS: Small image upscaled correctly")

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED!")
print("Bug #23 fix is working correctly in practice.")
