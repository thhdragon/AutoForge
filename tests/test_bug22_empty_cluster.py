"""
Test for Bug #22: Empty Cluster Handling in KMeans Splitting

This test verifies that empty sub-clusters during KMeans splitting
don't cause cluster ID misalignment.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
    init_height_map_depth_color_adjusted,
)


def test_empty_cluster_id_alignment():
    """
    Test that cluster IDs are sequential even when KMeans produces empty clusters.

    The bug occurs when:
    1. A cluster has high depth spread and gets split
    2. KMeans splitting produces an empty sub-cluster
    3. The empty cluster is skipped but new_cluster_id still increments
    4. This creates gaps in cluster IDs
    """
    # Create a simple image with clear depth separation
    # We'll create a scenario that can trigger empty clusters during splitting
    H, W = 20, 20

    # Create an image with two distinct regions
    target = np.zeros((H, W, 3), dtype=np.uint8)

    # Left half: dark color
    target[:, :10] = [51, 51, 51]  # 0.2 * 255

    # Right half: bright color
    target[:, 10:] = [204, 204, 204]  # 0.8 * 255

    # Create a depth map with high variance in one region
    # This should trigger cluster splitting
    depth_map = np.zeros((H, W), dtype=np.float32)

    # Create a cluster with high depth spread to trigger splitting
    # Left region: uniform depth
    depth_map[:, :10] = 0.5

    # Right region: high depth spread (should trigger split)
    # But make it such that KMeans might create empty clusters
    depth_map[:5, 10:] = 0.1  # Very shallow
    depth_map[5:, 10:] = 0.9  # Very deep

    # Run the initialization
    # Use fewer clusters to make the test more predictable
    num_clusters = 3
    _, height_map = init_height_map_depth_color_adjusted(
        target=target, max_layers=num_clusters, random_seed=42
    )

    # Verify the height map was created
    assert height_map is not None
    assert height_map.shape == (H, W)

    # Get unique cluster IDs
    unique_ids = np.unique(height_map)

    print(f"Unique cluster IDs: {sorted(unique_ids)}")
    print(f"Number of clusters: {len(unique_ids)}")

    # Check that cluster IDs are sequential (no gaps)
    # They should be 0, 1, 2, ... without gaps
    expected_ids = np.arange(len(unique_ids))

    # Sort both for comparison
    actual_ids_sorted = np.sort(unique_ids)

    # The bug causes gaps like [0, 1, 3] instead of [0, 1, 2]
    # After fix, we should have sequential IDs
    max_gap = np.max(np.diff(actual_ids_sorted)) if len(actual_ids_sorted) > 1 else 0

    print(f"Maximum gap between consecutive cluster IDs: {max_gap}")

    # After the fix, max gap should be 1 (sequential)
    # Before fix, it could be > 1 due to skipped IDs
    assert max_gap <= 1, (
        f"Cluster IDs have gaps: {actual_ids_sorted}, max gap = {max_gap}"
    )

    # Verify IDs start at 0
    assert actual_ids_sorted[0] == 0, (
        f"Cluster IDs should start at 0, got {actual_ids_sorted[0]}"
    )

    # Verify all cluster IDs are used (no missing IDs in sequence)
    for i in range(len(unique_ids)):
        assert i in unique_ids, f"Missing cluster ID {i} in sequence"

    print("✓ Cluster IDs are properly sequential")


def test_specific_empty_cluster_scenario():
    """
    Create a scenario that definitely triggers empty clusters during splitting.
    """
    # Very small image to make empty clusters more likely
    H, W = 10, 10

    # Create target with one dominant region (uint8 expected)
    target = np.ones((H, W, 3), dtype=np.uint8) * 128  # 0.5 * 255

    # Add a tiny different region (just a few pixels)
    target[0:2, 0:2] = [230, 230, 230]  # 0.9 * 255

    # Use more clusters than we have distinct regions
    # This increases likelihood of empty clusters
    _, height_map = init_height_map_depth_color_adjusted(
        target=target, max_layers=5, random_seed=123
    )

    assert height_map is not None
    unique_ids = np.unique(height_map)

    print(f"Scenario 2 - Unique cluster IDs: {sorted(unique_ids)}")

    # Check for sequential IDs
    for i in range(len(unique_ids)):
        assert i in unique_ids, f"Gap in cluster IDs at {i}"

    print("✓ No gaps in cluster IDs even with potential empty clusters")


def test_cluster_id_consistency():
    """
    Verify that the number of actual clusters matches the highest cluster ID + 1.
    """
    H, W = 15, 15

    # Random-ish pattern (uint8 expected)
    np.random.seed(42)
    target = (np.random.rand(H, W, 3) * 255).astype(np.uint8)

    # Run with enough clusters to trigger potential splitting
    pixel_height_logits, height_map = init_height_map_depth_color_adjusted(
        target=target, max_layers=4, random_seed=99
    )

    unique_ids = np.unique(height_map)
    num_clusters = len(unique_ids)
    max_id = np.max(unique_ids)

    print(f"Number of clusters: {num_clusters}")
    print(f"Max cluster ID: {max_id}")

    # After fix: max_id should be num_clusters - 1 (since IDs start at 0)
    # Before fix: max_id could be > num_clusters - 1 due to gaps
    assert max_id == num_clusters - 1, (
        f"Expected max ID {num_clusters - 1}, got {max_id}. IDs: {sorted(unique_ids)}"
    )

    print("✓ Cluster count and max ID are consistent")


if __name__ == "__main__":
    print("Testing Bug #22: Empty Cluster Handling\n")
    print("=" * 60)

    try:
        print("\nTest 1: Basic cluster ID alignment")
        print("-" * 60)
        test_empty_cluster_id_alignment()

        print("\n\nTest 2: Specific empty cluster scenario")
        print("-" * 60)
        test_specific_empty_cluster_scenario()

        print("\n\nTest 3: Cluster ID consistency")
        print("-" * 60)
        test_cluster_id_consistency()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
