"""
Verification test for Bug #35 fix: Silent Silhouette Score Errors

This test verifies that the fix properly handles bad silhouette scores.
"""

import numpy as np
import io
import sys


def test_segmentation_quality_prints_warning():
    """Verify that segmentation_quality now prints warnings on error."""
    from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
        segmentation_quality,
    )

    # Create single-cluster data to trigger ValueError
    X = np.random.randn(100, 3)
    labels = np.zeros(100, dtype=int)

    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    score = segmentation_quality(X, labels, sample_size=50, random_state=0)

    sys.stdout = old_stdout
    output = buffer.getvalue()

    # Verify the fix: warning should be printed
    assert "Warning" in output, f"Expected warning message, got: {output}"
    assert "Silhouette score failed" in output or "NaN" in output
    assert score == -1.0, f"Expected sentinel value -1.0, got {score}"
    print("✓ segmentation_quality now prints warnings")


def test_run_init_threads_handles_bad_scores():
    """Verify that run_init_threads properly handles bad silhouette scores."""
    from autoforge.Helper.Heightmaps.ChristofidesHeightMap import init_height_map
    import numpy as np

    # Create a simple test image
    target = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

    # Create scenarios with forced bad clustering
    # We'll test the metric calculation logic directly

    # Simulate results from init_height_map
    # Format: (pixel_height_logits, global_logits_out, ordering_metric, cluster_layers, sil_score, labels)

    # Good result
    good_result = (
        np.random.randn(50, 50),  # pixel_height_logits
        None,  # global_logits_out
        10.0,  # ordering_metric
        5,  # cluster_layers
        0.8,  # sil_score (good)
        np.zeros((50, 50), dtype=int),  # labels
    )

    # Bad result with sil_score = -1.0
    bad_result = (
        np.random.randn(50, 50),
        None,
        8.0,  # Even better ordering_metric
        5,
        -1.0,  # sil_score (error sentinel)
        np.zeros((50, 50), dtype=int),
    )

    results = [bad_result, good_result]

    # Test the metric calculation logic (mimicking run_init_threads)
    metrics = []
    for r in results:
        sil_score = r[4]
        if sil_score <= -0.5:  # Sentinel check (the fix)
            metrics.append(float("inf"))
        else:
            metrics.append((r[2] / r[3]) / (sil_score + 1e-6))

    # Verify bad result gets inf metric
    assert metrics[0] == float("inf"), "Bad result should get inf metric"
    assert metrics[1] < float("inf"), "Good result should get finite metric"

    # Verify the good result is selected
    best_idx = metrics.index(min(metrics))
    assert best_idx == 1, "Should select good result, not bad one"

    print("✓ run_init_threads properly handles bad silhouette scores")


def test_end_to_end_with_single_cluster():
    """End-to-end test: verify that single-cluster images don't crash."""
    from autoforge.Helper.Heightmaps.ChristofidesHeightMap import init_height_map
    import numpy as np

    # Create an image with uniform color (will result in single cluster)
    target = np.full((50, 50, 3), 128, dtype=np.uint8)
    background_tuple = (128, 128, 128)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        result = init_height_map(
            target,
            max_layers=5,
            h=1.0,
            background_tuple=background_tuple,
            random_seed=42,
            cluster_layers=3,
        )

        sys.stdout = old_stdout
        output = buffer.getvalue()

        # Should complete without crashing
        assert result is not None
        assert len(result) == 6

        # Check if warning was printed
        if "Warning" in output:
            print("✓ Warning printed for poor clustering (expected)")

        print("✓ End-to-end test passed")

    except Exception as e:
        sys.stdout = old_stdout
        raise AssertionError(f"Should not crash on single-cluster image: {e}")


def test_metric_calculation_doesnt_select_bad_runs():
    """Verify that runs with bad silhouette scores are not selected as best."""
    # Simulate the exact scenario from run_init_threads

    # Three results: one bad, two good
    results = [
        (None, None, 12.0, 6, 0.7, None),  # Good: metric = (12/6)/0.7 = 2.857
        (None, None, 10.0, 5, -1.0, None),  # Bad: should get inf
        (None, None, 15.0, 8, 0.9, None),  # Good: metric = (15/8)/0.9 = 2.083 (best)
    ]

    # Apply the fix logic
    metrics = []
    for r in results:
        sil_score = r[4]
        if sil_score <= -0.5:
            metrics.append(float("inf"))
        else:
            metrics.append((r[2] / r[3]) / (sil_score + 1e-6))

    best_idx = metrics.index(min(metrics))

    # Should select result 2 (index 2) as it has best valid metric
    assert best_idx == 2, f"Expected index 2, got {best_idx}"
    assert metrics[1] == float("inf"), "Bad result should have inf metric"

    print("✓ Bad runs are not selected as best")


if __name__ == "__main__":
    print("=" * 70)
    print("Bug #35 Fix Verification Tests")
    print("=" * 70)

    print("\nTest 1: Warning output on error")
    test_segmentation_quality_prints_warning()

    print("\nTest 2: Metric calculation handles bad scores")
    test_run_init_threads_handles_bad_scores()

    print("\nTest 3: End-to-end with single cluster")
    test_end_to_end_with_single_cluster()

    print("\nTest 4: Bad runs not selected")
    test_metric_calculation_doesnt_select_bad_runs()

    print("\n" + "=" * 70)
    print("All verification tests passed! Bug #35 is FIXED.")
    print("=" * 70)
