# Bug #11 Fix Report: Coplanar Smoothing Dimension Swap

**Status**: ✅ FIXED  
**Severity**: 🔴 CRITICAL  
**Date Fixed**: December 2025

---

## Bug Summary

In the `smooth_coplanar_faces()` function in [PruningHelper.py](PruningHelper.py#L768-L834), the dimensions for shifting the surface normals tensor were reversed. This caused incorrect neighbor sampling during coplanar smoothing, leading to smoothing artifacts and wrong height map adjustments.

### Root Cause

The normals tensor has shape `(3, H, W)` where:

- **dim=0**: Normal components (x, y, z)
- **dim=1**: HEIGHT dimension
- **dim=2**: WIDTH dimension

When iterating through 8 neighbors with shifts `(dx, dy)`:

- `dx` represents a shift along the **WIDTH** axis (columns)
- `dy` represents a shift along the **HEIGHT** axis (rows)

The buggy code incorrectly shifted:

```python
neighbor_normals = torch.roll(
    torch.roll(normals, shifts=dx, dims=1),  # ❌ WRONG: shifts dx on HEIGHT
    shifts=dy, dims=2  # ❌ WRONG: shifts dy on WIDTH
)
```

This resulted in the wrong neighbors being selected for coplanar angle checks.

---

## The Fix

Changed lines 810-811 in [PruningHelper.py](PruningHelper.py#L810-L811) to correct the dimension mapping:

```python
# BEFORE (Buggy):
neighbor_normals = torch.roll(
    torch.roll(normals, shifts=dx, dims=1), shifts=dy, dims=2
)

# AFTER (Fixed):
neighbor_normals = torch.roll(
    torch.roll(normals, shifts=dy, dims=1), shifts=dx, dims=2
)
```

Also added a clarifying comment to explain the mapping.

---

## Verification

### Test Results

1. **Dimension swap verification test** ([test_bug11_dimension_swap.py](../tests/test_bug11_dimension_swap.py)):
   - ✅ Confirms buggy and fixed versions produce different results
   - ✅ Validates correct neighbor selection
   - ✅ Tests gradient direction correctness

2. **Fixed version validation test** ([test_bug11_fixed_validation.py](../tests/test_bug11_fixed_validation.py)):
   - ✅ Fixed version produces correct smoothing
   - ✅ Preserves planar surfaces correctly
   - ✅ Processes gradient surfaces without errors

3. **Regression tests**:
   - ✅ `test_pruning_helper.py`: 3/3 passed
   - ✅ `test_output_helper.py`: 4/4 passed
   - ✅ No existing tests broken

### Impact Analysis

**Before Fix**:

- Wrong neighbors selected → smoothing applied to incorrect pixels
- Coplanar detection unreliable → height map distortions
- Artifacts in surface smoothing

**After Fix**:

- Correct neighbors selected → proper coplanar detection
- Height maps smoothed correctly → better visual quality
- Surfaces properly preserved according to angle thresholds

---

## Code Quality Improvements

Added clarifying comment to document the dimension mapping:

```python
# dx shifts along WIDTH (columns, dim=2), dy shifts along HEIGHT (rows, dim=1)
```

---

## Files Modified

- [src/autoforge/Helper/PruningHelper.py](PruningHelper.py#L810-L811) - Fixed dimension swap

## Tests Added

- [tests/test_bug11_dimension_swap.py](../tests/test_bug11_dimension_swap.py) - Bug verification
- [tests/test_bug11_fixed_validation.py](../tests/test_bug11_fixed_validation.py) - Fix validation

---

## Testing Instructions

To verify the fix:

```bash
# Run bug verification tests
python -m pytest tests/test_bug11_dimension_swap.py -v

# Run fix validation tests
python -m pytest tests/test_bug11_fixed_validation.py -v

# Run all related tests
python -m pytest tests/test_pruning_helper.py tests/test_output_helper.py -v
```

All tests pass ✅

---

## Physics/Algorithm Explanation

The `smooth_coplanar_faces()` function:

1. Computes surface normals via finite differences
2. For each pixel, examines 8 neighbors
3. Compares normal angles to determine if neighbors are coplanar
4. Averages the pixel with coplanar neighbors

**The Fix Ensures**:

- Each neighbor is correctly identified spatially
- Angle comparisons are made between actual neighboring normals
- Smoothing is applied to the right pixels

This is essential for 3D printing where coplanar surfaces should remain flat and smooth.

---

## Summary

Bug #11 was a critical dimension mismatch in the coplanar smoothing function that caused incorrect neighbor sampling. The fix swaps the shift parameters to correctly map spatial offsets to tensor dimensions. All tests pass and the fix has no negative impact on existing functionality.

**Priority**: FIXED ✅  
**Risk Level**: LOW (isolated change, well-tested)  
**Recommended Action**: Merge immediately
