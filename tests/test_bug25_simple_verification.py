"""
Simplified test for Bug 25 Fix: Test the clustering logic directly

This test verifies the fix without requiring the full depth estimation pipeline.
"""

import numpy as np
from sklearn.cluster import KMeans


def test_bug25_fix_clustering_logic():
    """
    Test the actual clustering logic fix in isolation.

    Simulates what happens in DepthEstimateHeightMap.py lines 148-156.
    """
    print("=" * 70)
    print("BUG 25 FIX: Testing Clustering Logic")
    print("=" * 70)

    test_cases = [
        # (height, width, max_layers, expected_behavior)
        (10, 10, 75, "Should limit clusters to avoid over-clustering"),
        (5, 5, 100, "Should limit clusters for very small image"),
        (100, 100, 75, "Should use max_layers for normal image"),
        (50, 50, 75, "Should use max_layers for medium image"),
    ]

    for h, w, max_layers, description in test_cases:
        print(f"\n{description}")
        print(f"Image: {h}x{w}, max_layers: {max_layers}")

        # Create random test image
        np.random.seed(42)
        target = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        pixels = target.reshape(-1, 3).astype(np.float32)

        total_pixels = len(pixels)

        # OLD (BUGGY) CODE:
        optimal_n_old = max_layers

        # NEW (FIXED) CODE:
        unique_pixels = np.unique(pixels, axis=0)
        max_reasonable_clusters = min(max_layers, len(pixels) // 2, len(unique_pixels))
        optimal_n_new = max(2, max_reasonable_clusters)

        print(f"  Total pixels: {total_pixels}")
        print(f"  Unique colors: {len(unique_pixels)}")
        print(
            f"  OLD optimal_n: {optimal_n_old} → {total_pixels / optimal_n_old:.2f} pixels/cluster"
        )
        print(
            f"  NEW optimal_n: {optimal_n_new} → {total_pixels / optimal_n_new:.2f} pixels/cluster"
        )

        # Verify the fix
        if total_pixels < max_layers * 2:  # Small image case
            assert optimal_n_new < optimal_n_old, (
                "Fix should reduce clusters for small images"
            )
            assert optimal_n_new <= total_pixels // 2, "Should not over-cluster"
        else:  # Normal image case
            assert optimal_n_new <= max_layers, "Should not exceed max_layers"

        assert optimal_n_new >= 2, "Should always have at least 2 clusters"
        assert total_pixels / optimal_n_new >= 2.0, (
            "Should have reasonable cluster sizes"
        )

        print(f"  ✓ Fix works correctly!")

    print("\n" + "=" * 70)
    print("All clustering logic tests passed!")
    print("=" * 70)


def test_bug25_actual_kmeans_behavior():
    """
    Test actual KMeans behavior with the fix to ensure it works in practice.
    """
    print("\n" + "=" * 70)
    print("BUG 25 FIX: Testing Actual KMeans Clustering")
    print("=" * 70)

    # Small image case
    np.random.seed(42)
    small_image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    pixels = small_image.reshape(-1, 3).astype(np.float32)
    max_layers = 75

    print(f"\nSmall Image Test: 10x10 with max_layers={max_layers}")

    # Apply fix
    unique_pixels = np.unique(pixels, axis=0)
    max_reasonable_clusters = min(max_layers, len(pixels) // 2, len(unique_pixels))
    optimal_n = max(2, max_reasonable_clusters)

    print(f"  Clustering with optimal_n={optimal_n}")

    # Run KMeans
    labels = KMeans(n_clusters=optimal_n, random_state=42).fit_predict(pixels)

    # Analyze results
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"  Clusters created: {len(unique_labels)}")
    print(f"  Min pixels/cluster: {counts.min()}")
    print(f"  Max pixels/cluster: {counts.max()}")
    print(f"  Mean pixels/cluster: {counts.mean():.2f}")
    print(f"  Clusters with <=2 pixels: {np.sum(counts <= 2)}")

    # Verify reasonable clustering
    assert len(unique_labels) == optimal_n, "Should create expected number of clusters"
    assert counts.mean() >= 2.0, "Should have reasonable cluster sizes"
    assert optimal_n <= 50, "Small image should have limited clusters"

    print("  ✓ KMeans clustering works correctly with fix!")

    # Normal image case
    print(f"\nNormal Image Test: 100x100 with max_layers={max_layers}")

    np.random.seed(42)
    normal_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    pixels = normal_image.reshape(-1, 3).astype(np.float32)

    # Apply fix
    unique_pixels = np.unique(pixels, axis=0)
    max_reasonable_clusters = min(max_layers, len(pixels) // 2, len(unique_pixels))
    optimal_n = max(2, max_reasonable_clusters)

    print(f"  Clustering with optimal_n={optimal_n}")

    # Run KMeans
    labels = KMeans(n_clusters=optimal_n, random_state=42).fit_predict(pixels)

    # Analyze results
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"  Clusters created: {len(unique_labels)}")
    print(f"  Min pixels/cluster: {counts.min()}")
    print(f"  Max pixels/cluster: {counts.max()}")
    print(f"  Mean pixels/cluster: {counts.mean():.2f}")

    # Verify reasonable clustering
    assert len(unique_labels) == optimal_n, "Should create expected number of clusters"
    assert optimal_n == max_layers, "Normal image should use max_layers"
    assert counts.mean() > 100, "Normal image should have many pixels per cluster"

    print("  ✓ Normal image clustering works correctly!")

    print("\n" + "=" * 70)
    print("All KMeans tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_bug25_fix_clustering_logic()
    test_bug25_actual_kmeans_behavior()
    print("\n✅ Bug 25 fix verified successfully!")
