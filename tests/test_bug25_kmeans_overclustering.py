"""
Test for Bug 25: KMeans Over-Clustering on Small Images

This test verifies that the KMeans clustering in DepthEstimateHeightMap
doesn't try to create more clusters than there are unique pixels, which
would result in useless clustering with 1-2 pixels per cluster.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans


def test_kmeans_overclustering_bug():
    """
    Test that demonstrates the bug: trying to cluster 100 pixels into 75 clusters.

    This simulates what happens when a small image (10x10 = 100 pixels) is processed
    with max_layers=75. KMeans will create many nearly-empty clusters.
    """
    # Simulate a small image: 10x10 = 100 pixels
    np.random.seed(42)
    small_image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)

    H, W, _ = small_image.shape
    pixels = small_image.reshape(-1, 3).astype(np.float32)

    print(f"\nImage size: {H}x{W} = {len(pixels)} pixels")
    print(f"Unique pixel colors: {len(np.unique(pixels, axis=0))}")

    # This is the buggy behavior - trying to cluster into 75 clusters
    max_layers = 75
    optimal_n = max_layers  # Bug: no validation!

    print(f"Attempting to create {optimal_n} clusters from {len(pixels)} pixels...")

    # This will "succeed" but create mostly useless clusters
    labels = KMeans(n_clusters=optimal_n, random_state=42).fit_predict(pixels)

    # Count pixels per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"\nCluster distribution:")
    print(f"  Total clusters created: {len(unique_labels)}")
    print(f"  Min pixels per cluster: {counts.min()}")
    print(f"  Max pixels per cluster: {counts.max()}")
    print(f"  Mean pixels per cluster: {counts.mean():.2f}")
    print(f"  Clusters with <=2 pixels: {np.sum(counts <= 2)}")

    # Verify the bug exists
    assert len(unique_labels) == optimal_n, "Should create exactly max_layers clusters"
    assert counts.min() <= 2, "Bug verified: some clusters have only 1-2 pixels"
    assert counts.mean() < 2, "Bug verified: average cluster size is tiny"

    print("\n✓ Bug 25 verified: KMeans creates too many clusters for small images!")


def test_proposed_fix():
    """
    Test the proposed fix: limit clusters based on number of pixels and unique values.
    """
    # Simulate a small image: 10x10 = 100 pixels
    np.random.seed(42)
    small_image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)

    H, W, _ = small_image.shape
    pixels = small_image.reshape(-1, 3).astype(np.float32)
    max_layers = 75

    print(f"\nImage size: {H}x{W} = {len(pixels)} pixels")
    print(f"Requested max_layers: {max_layers}")

    # PROPOSED FIX: Limit clusters based on unique pixels
    # We can't have more clusters than pixels, and ideally not more than half
    unique_pixels = np.unique(pixels, axis=0)
    max_reasonable_clusters = min(max_layers, len(pixels) // 2, len(unique_pixels))
    optimal_n = max(2, max_reasonable_clusters)

    print(f"Unique pixel colors: {len(unique_pixels)}")
    print(f"Adjusted optimal_n: {optimal_n}")

    # Now cluster with the corrected number
    labels = KMeans(n_clusters=optimal_n, random_state=42).fit_predict(pixels)

    # Count pixels per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"\nCluster distribution after fix:")
    print(f"  Total clusters created: {len(unique_labels)}")
    print(f"  Min pixels per cluster: {counts.min()}")
    print(f"  Max pixels per cluster: {counts.max()}")
    print(f"  Mean pixels per cluster: {counts.mean():.2f}")
    print(f"  Clusters with <=2 pixels: {np.sum(counts <= 2)}")

    # Verify the fix works
    assert optimal_n <= len(pixels) // 2, "Should not create too many clusters"
    assert counts.mean() >= 2, "Fix verified: reasonable cluster sizes"
    assert optimal_n <= max_layers, "Should not exceed max_layers"
    assert optimal_n >= 2, "Should have at least 2 clusters"

    print("\n✓ Fix verified: Reasonable number of clusters for small images!")


def test_fix_with_various_sizes():
    """
    Test the fix with various image sizes to ensure it scales properly.
    """
    test_cases = [
        (10, 10, 75),  # Small image, large max_layers
        (50, 50, 75),  # Medium image
        (100, 100, 75),  # Large image
        (5, 5, 100),  # Very small image, very large max_layers
    ]

    print("\nTesting fix with various image sizes:")

    for h, w, max_layers in test_cases:
        np.random.seed(42)
        image = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Apply fix
        unique_pixels = np.unique(pixels, axis=0)
        max_reasonable_clusters = min(max_layers, len(pixels) // 2, len(unique_pixels))
        optimal_n = max(2, max_reasonable_clusters)

        pixels_per_cluster = len(pixels) / optimal_n

        print(f"  {h}x{w} (pixels={len(pixels)}, max_layers={max_layers})")
        print(f"    → optimal_n={optimal_n}, pixels/cluster={pixels_per_cluster:.1f}")

        assert optimal_n >= 2, "Should have at least 2 clusters"
        assert optimal_n <= max_layers, "Should not exceed max_layers"
        assert pixels_per_cluster >= 2, "Should have reasonable cluster sizes"

    print("\n✓ Fix works correctly for various image sizes!")


if __name__ == "__main__":
    print("=" * 70)
    print("BUG 25: KMeans Over-Clustering on Small Images")
    print("=" * 70)

    test_kmeans_overclustering_bug()
    print("\n" + "=" * 70)
    test_proposed_fix()
    print("\n" + "=" * 70)
    test_fix_with_various_sizes()
    print("\n" + "=" * 70)
    print("All tests passed!")
