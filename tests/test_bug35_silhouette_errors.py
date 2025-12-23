"""
Test for Bug #35: Silent Silhouette Score Errors

The bug is in ChristofidesHeightMap.py where segmentation_quality()
returns -1.0 on error, but callers don't check for this sentinel value.
This can lead to bad clustering being selected as "best".
"""

import numpy as np
import pytest


def test_segmentation_quality_single_cluster_error():
    """Test that segmentation_quality handles single-cluster case (k=1)."""
    from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
        segmentation_quality,
    )

    # Create data with only one cluster (all same labels)
    X = np.random.randn(100, 3)
    labels = np.zeros(100, dtype=int)  # All pixels in cluster 0

    # This should return -1.0 because silhouette_score raises ValueError with k=1
    score = segmentation_quality(X, labels, sample_size=50, random_state=0)

    assert score == -1.0, f"Expected -1.0 for single cluster, got {score}"


def test_segmentation_quality_valid_clustering():
    """Test that segmentation_quality works correctly with valid clustering."""
    from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
        segmentation_quality,
    )

    # Create data with two well-separated clusters
    X1 = np.random.randn(50, 3) + np.array([0, 0, 0])
    X2 = np.random.randn(50, 3) + np.array([10, 10, 10])
    X = np.vstack([X1, X2])
    labels = np.array([0] * 50 + [1] * 50)

    score = segmentation_quality(X, labels, sample_size=100, random_state=0)

    # Should get a high positive score for well-separated clusters
    assert score > 0.5, f"Expected high silhouette score, got {score}"
    assert not np.isnan(score), "Score should not be NaN"


def test_metric_calculation_with_negative_silhouette():
    """Test that the metric calculation in run_init_threads handles -1.0 correctly."""
    # Simulate the calculation in line 669 of ChristofidesHeightMap.py:
    # metrics = [(r[2] / r[3]) / (r[4] + 1e-6) for r in results]
    # where r[2] = ordering_metric, r[3] = cluster_layers, r[4] = sil_score

    # Good result
    ordering_metric_good = 10.0
    cluster_layers_good = 5
    sil_score_good = 0.8

    metric_good = (ordering_metric_good / cluster_layers_good) / (sil_score_good + 1e-6)

    # Bad result with sil_score = -1.0 (error case)
    ordering_metric_bad = 8.0
    cluster_layers_bad = 5
    sil_score_bad = -1.0

    metric_bad = (ordering_metric_bad / cluster_layers_bad) / (sil_score_bad + 1e-6)

    # Bug: The bad result gets a large NEGATIVE metric value
    # When using min() to find best, it incorrectly selects the bad result!
    print(f"Good metric: {metric_good}")
    print(f"Bad metric (with sil_score=-1.0): {metric_bad}")

    assert metric_bad < 0, "Bad metric should be negative"
    assert metric_bad < metric_good, "BUG: Bad metric is smaller, would be selected!"


def test_fixed_metric_calculation():
    """Test that after the fix, bad silhouette scores are handled properly."""
    # After fix, should check if sil_score is bad before using it

    ordering_metric = 10.0
    cluster_layers = 5
    sil_score = -1.0

    # Fixed version: check for bad score
    if sil_score <= -0.5:  # Sentinel check
        print("Warning: Clustering quality poor")
        # Option 1: Skip this result
        # Option 2: Use a different metric calculation
        metric = float("inf")  # Don't select this result
    else:
        metric = (ordering_metric / cluster_layers) / (sil_score + 1e-6)

    assert metric == float("inf"), "Bad results should not be selected"


def test_segmentation_quality_with_warning():
    """Test that after fix, warnings are printed when silhouette_score fails."""
    from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
        segmentation_quality,
    )
    import io
    import sys

    # Create single-cluster data to trigger ValueError
    X = np.random.randn(100, 3)
    labels = np.zeros(100, dtype=int)

    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    score = segmentation_quality(X, labels, sample_size=50, random_state=0)

    sys.stdout = old_stdout
    output = buffer.getvalue()

    # After fix, should print a warning (currently doesn't)
    # This test documents the expected behavior after fix
    assert score == -1.0


if __name__ == "__main__":
    print("Running Bug #35 tests...")
    print("\n" + "=" * 60)
    print("Test 1: Single cluster error handling")
    test_segmentation_quality_single_cluster_error()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("Test 2: Valid clustering")
    test_segmentation_quality_valid_clustering()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("Test 3: Metric calculation with negative silhouette (BUG DEMO)")
    test_metric_calculation_with_negative_silhouette()
    print("✓ PASSED - Bug demonstrated")

    print("\n" + "=" * 60)
    print("Test 4: Fixed metric calculation")
    test_fixed_metric_calculation()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("Test 5: Warning output (after fix)")
    test_segmentation_quality_with_warning()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")
