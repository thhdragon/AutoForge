"""
Test for floating point precision issues
"""

test_cases = [
    (101, 50),
    (103, 50),
    (107, 50),
    (109, 50),
    (113, 50),
    (997, 498),
    (1001, 500),
]

print("Testing for floating point precision issues:")
print("=" * 60)

for dim, max_size in test_cases:
    scale = max_size / dim
    result = dim * scale
    rounded = int(round(result))

    exact_match = result == max_size
    rounded_match = rounded == max_size

    status = "✓" if rounded_match else "✗"

    print(f"dim={dim:4}, max_size={max_size:3}, scale={scale:.15f}")
    print(f"  dim * scale = {result:.15f}")
    print(f"  round(dim * scale) = {rounded}")
    print(f"  {status} Matches max_size: exact={exact_match}, rounded={rounded_match}")
    print()
