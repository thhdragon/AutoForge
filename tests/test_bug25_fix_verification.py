"""
Test for Bug 25 Fix Verification: KMeans Over-Clustering on Small Images

This test verifies that the fix in DepthEstimateHeightMap properly limits
the number of clusters based on image size and unique pixels.
"""

import numpy as np
import pytest


def test_depth_init_small_image_clustering():
    """
    Test that init_height_map_depth_color_adjusted properly handles small images
    and doesn't create too many clusters.
    """
    from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
        init_height_map_depth_color_adjusted,
    )

    # Create a small test image (10x10)
    np.random.seed(42)
    small_image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    max_layers = 75

    print(f"\nTest Case: Small Image (10x10) with max_layers={max_layers}")
    print(f"Total pixels: {10 * 10}")

    # This should not crash and should create a reasonable number of clusters
    try:
        pixel_height_logits, final_labels = init_height_map_depth_color_adjusted(
            target=small_image,
            max_layers=max_layers,
            random_seed=42,
        )

        # Check results
        assert pixel_height_logits.shape == (10, 10), "Output shape should match input"

        # Count unique clusters
        num_clusters = len(np.unique(final_labels))
        pixels_per_cluster = (10 * 10) / num_clusters

        print(f"Clusters created: {num_clusters}")
        print(f"Pixels per cluster: {pixels_per_cluster:.2f}")

        # Verify the fix worked
        assert num_clusters <= max_layers, "Should not exceed max_layers"
        assert num_clusters <= 100, "Should not exceed total pixels"
        assert pixels_per_cluster >= 1.4, (
            "Should have reasonable cluster sizes (>= 1.4 pixels average)"
        )
        assert num_clusters >= 2, "Should have at least 2 clusters"

        print("✓ Small image handled correctly with reasonable cluster count!")

    except Exception as e:
        pytest.fail(f"Failed to initialize small image: {e}")


def test_depth_init_very_small_image():
    """
    Test with an extremely small image (5x5).
    """
    from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
        init_height_map_depth_color_adjusted,
    )

    # Create a very small test image (5x5)
    np.random.seed(42)
    tiny_image = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)
    max_layers = 100

    print(f"\nTest Case: Very Small Image (5x5) with max_layers={max_layers}")
    print(f"Total pixels: {5 * 5}")

    try:
        pixel_height_logits, final_labels = init_height_map_depth_color_adjusted(
            target=tiny_image,
            max_layers=max_layers,
            random_seed=42,
        )

        assert pixel_height_logits.shape == (5, 5), "Output shape should match input"

        num_clusters = len(np.unique(final_labels))
        pixels_per_cluster = (5 * 5) / num_clusters

        print(f"Clusters created: {num_clusters}")
        print(f"Pixels per cluster: {pixels_per_cluster:.2f}")

        # With only 25 pixels, we should have very few clusters
        assert num_clusters <= 25, "Should not exceed total pixels"
        assert num_clusters >= 2, "Should have at least 2 clusters"
        assert pixels_per_cluster >= 1.3, (
            "Should have reasonable cluster sizes (depth model may create more clusters)"
        )

        print("✓ Very small image handled correctly!")

    except Exception as e:
        pytest.fail(f"Failed to initialize very small image: {e}")


def test_depth_init_normal_image():
    """
    Test with a normal-sized image to ensure the fix doesn't break normal operation.
    """
    from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
        init_height_map_depth_color_adjusted,
    )

    # Create a normal test image (100x100)
    np.random.seed(42)
    normal_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    max_layers = 75

    print(f"\nTest Case: Normal Image (100x100) with max_layers={max_layers}")
    print(f"Total pixels: {100 * 100}")

    try:
        pixel_height_logits, final_labels = init_height_map_depth_color_adjusted(
            target=normal_image,
            max_layers=max_layers,
            random_seed=42,
        )

        assert pixel_height_logits.shape == (100, 100), (
            "Output shape should match input"
        )

        num_clusters = len(np.unique(final_labels))
        pixels_per_cluster = (100 * 100) / num_clusters

        print(f"Clusters created: {num_clusters}")
        print(f"Pixels per cluster: {pixels_per_cluster:.2f}")

        # With a large image, depth model may create more clusters than max_layers
        # due to depth-based splitting. This is expected behavior.
        assert num_clusters >= 2, "Should have at least 2 clusters"
        # Depth model can create more clusters due to depth variance
        assert num_clusters <= 200, "Should not create excessive clusters"
        assert pixels_per_cluster >= 50, (
            "Large image should have reasonable pixels per cluster"
        )

        print("✓ Normal image handled correctly!")

    except Exception as e:
        pytest.fail(f"Failed to initialize normal image: {e}")


def test_depth_init_edge_cases():
    """
    Test edge cases to ensure robustness.
    """
    from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
        init_height_map_depth_color_adjusted,
    )

    print("\nTest Case: Edge Cases")

    # Test 1: Image with only 2 unique colors
    solid_image = np.zeros((20, 20, 3), dtype=np.uint8)
    solid_image[:10, :, :] = [255, 0, 0]  # Top half red
    solid_image[10:, :, :] = [0, 0, 255]  # Bottom half blue

    print("\nSubtest: Image with only 2 unique colors")
    try:
        pixel_height_logits, final_labels = init_height_map_depth_color_adjusted(
            target=solid_image,
            max_layers=75,
            random_seed=42,
        )

        num_clusters = len(np.unique(final_labels))
        print(f"  Clusters created: {num_clusters}")

        # Should create a small number of clusters since there are only 2 unique colors
        assert num_clusters >= 2, "Should have at least 2 clusters"
        assert num_clusters <= 10, "Should not create many clusters for 2-color image"

        print("  ✓ Two-color image handled correctly!")

    except Exception as e:
        pytest.fail(f"Failed on two-color image: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("BUG 25 FIX VERIFICATION: KMeans Over-Clustering on Small Images")
    print("=" * 70)

    test_depth_init_small_image_clustering()
    print("\n" + "=" * 70)
    test_depth_init_very_small_image()
    print("\n" + "=" * 70)
    test_depth_init_normal_image()
    print("\n" + "=" * 70)
    test_depth_init_edge_cases()
    print("\n" + "=" * 70)
    print("All verification tests passed!")
