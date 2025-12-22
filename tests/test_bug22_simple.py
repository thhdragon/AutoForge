"""
Simple test for Bug #22: Verify cluster ID handling in KMeans splitting

This test directly examines the clustering logic to see if empty clusters
cause ID gaps.
"""

import numpy as np
from sklearn.cluster import KMeans


def simulate_cluster_splitting():
    """
    Simulate the exact logic from DepthEstimateHeightMap.py lines 157-173
    to see if empty clusters can cause ID gaps.
    """
    # Simulate having 3 initial clusters with varying depth spreads    np.random.seed(42)

    # Create labels (simulating Step 3 output)
    H, W = 20, 20
    labels = np.zeros((H, W), dtype=int)
    labels[:, :7] = 0  # Cluster 0
    labels[:, 7:14] = 1  # Cluster 1
    labels[:, 14:] = 2  # Cluster 2

    # Create depth map where cluster 1 has high variance (will split)
    depth_norm = np.random.rand(H, W) * 0.1 + 0.5  # Low variance baseline
    # Make cluster 1 have high depth spread
    depth_norm[:10, 7:14] = 0.1  # Half at low depth
    depth_norm[10:, 7:14] = 0.9  # Half at high depth

    depth_threshold = 0.2
    random_seed = 42

    # Simulate the clustering logic from lines 157-173
    final_labels = np.copy(labels)
    new_cluster_id = 0
    cluster_info = {}
    unique_labels = np.unique(labels)

    print("Processing clusters:")
    for orig_label in unique_labels:
        mask = labels == orig_label
        cluster_depths = depth_norm[mask]
        avg_depth = np.mean(cluster_depths)
        depth_range = cluster_depths.max() - cluster_depths.min()

        print(
            f"  Cluster {orig_label}: depth_range={depth_range:.3f}, pixels={np.sum(mask)}"
        )

        if depth_range > depth_threshold:
            print(f"    → Splitting cluster {orig_label}")
            # Split this cluster
            depth_values = cluster_depths.reshape(-1, 1)
            k_split = 2
            kmeans_split = KMeans(n_clusters=k_split, random_state=random_seed)
            split_labels = kmeans_split.fit_predict(depth_values)
            indices = np.argwhere(mask)

            for split in range(k_split):
                sub_mask = split_labels == split
                inds = indices[sub_mask]
                print(f"      Sub-cluster {split}: {inds.size} pixels")

                if inds.size == 0:
                    print(f"        → Empty! Skipping (ID {new_cluster_id} not used)")
                    continue

                for i, j in inds:
                    final_labels[i, j] = new_cluster_id
                sub_avg_depth = np.mean(depth_norm[mask][split_labels == split])
                cluster_info[new_cluster_id] = sub_avg_depth
                print(f"        → Assigned ID {new_cluster_id}")
                new_cluster_id += 1
        else:
            print(f"    → Not splitting")
            indices = np.argwhere(mask)
            for i, j in indices:
                final_labels[i, j] = new_cluster_id
            cluster_info[new_cluster_id] = avg_depth
            print(f"    → Assigned ID {new_cluster_id}")
            new_cluster_id += 1

    num_final_clusters = new_cluster_id

    print(f"\nFinal results:")
    print(f"  num_final_clusters = {num_final_clusters}")
    print(f"  cluster_info keys = {sorted(cluster_info.keys())}")
    print(f"  unique labels in final_labels = {sorted(np.unique(final_labels))}")

    # Check for gaps
    expected_ids = list(range(num_final_clusters))
    actual_ids = sorted(cluster_info.keys())

    print(f"\nChecking for gaps:")
    print(f"  Expected IDs: {expected_ids}")
    print(f"  Actual IDs in cluster_info: {actual_ids}")

    if expected_ids == actual_ids:
        print("  ✓ No gaps in cluster IDs")
        return True
    else:
        missing = set(expected_ids) - set(actual_ids)
        print(f"  ✗ Missing IDs: {missing}")
        return False


def test_forced_empty_cluster():
    """
    Try to force an empty cluster by using a very skewed distribution.
    """
    print("\n" + "=" * 60)
    print("Testing forced empty cluster scenario")
    print("=" * 60)

    # Create a scenario where KMeans k=2 will likely create an empty cluster
    # Use a very small cluster with just a few identical values
    depth_values = np.array([[0.5]] * 5 + [[0.51]] * 2).reshape(-1, 1)

    print(f"Depth values: {depth_values.flatten()}")

    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(depth_values)

    print(f"KMeans labels: {labels}")
    print(f"Label 0 count: {np.sum(labels == 0)}")
    print(f"Label 1 count: {np.sum(labels == 1)}")

    # Check if we got an empty cluster
    if np.sum(labels == 0) == 0 or np.sum(labels == 1) == 0:
        print("✓ Successfully created an empty cluster!")
        return True
    else:
        print("✗ Could not create an empty cluster in this scenario")
        return False


if __name__ == "__main__":
    print("Bug #22 Investigation: Empty Cluster Handling\n")

    success = simulate_cluster_splitting()

    # Try to force empty clusters
    test_forced_empty_cluster()

    print("\n" + "=" * 60)
    if success:
        print("CONCLUSION: Code appears correct - no gaps detected")
        print("The bug may have been fixed already, or the bug")
        print("description may be inaccurate.")
    else:
        print("CONCLUSION: Bug confirmed - gaps in cluster IDs detected!")
    print("=" * 60)
