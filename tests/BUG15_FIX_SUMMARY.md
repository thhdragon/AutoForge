# Bug #15 Fix Summary

**Status**: ✅ FIXED  
**Date**: December 22, 2025  
**Severity**: HIGH  
**Category**: Physics/Numerical

---

## Problem

The opacity formula had a **negative linear term** that caused opacity to **decrease** at high thickness ratios, which is physically incorrect.

### Original Buggy Formula

```python
o, A, k, b = -1.2416557e-02, 9.6407950e-01, 3.4103447e01, -4.1554203e00
opac = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
opac = torch.clamp(opac, 0.0, 1.0)
```

### Issues with Original Formula

1. **Opacity peaked at thick_ratio ≈ 0.12** then **decreased to 0** by thick_ratio = 1.0
2. At thick_ratio = 1.0 (when thickness equals TD), opacity was **0%** (nonsensical!)
3. **138 points** where opacity was decreasing (physically impossible)
4. **Zero gradients** at thick_ratio ≥ 1.0 (optimization cannot work)
5. Violated basic physics: thicker materials should be more opaque, not less

### Test Results (Before Fix)

```
thick_ratio=0.0: opacity=0.0000  ✓ OK
thick_ratio=0.1: opacity=0.9997  ⚠️ Already saturated!
thick_ratio=0.5: opacity=0.7027  ⚠️ Decreasing
thick_ratio=1.0: opacity=0.0000  ✗ WRONG!
thick_ratio=2.0: opacity=0.0000  ✗ WRONG!
thick_ratio=3.0: opacity=0.0000  ✗ WRONG!
```

---

## Solution

Replace with **Beer-Lambert law** based formula:

```python
# Bug #15 Fix: Use Beer-Lambert law for physically correct opacity
# opacity = 1 - exp(-k * thick_ratio)
# where thick_ratio = thickness / TD
# k=2.5 provides good saturation: opacity≈0.92 at thick_ratio=1.0
k_opacity = 2.5
opac = 1.0 - torch.exp(-k_opacity * thick_ratio)  # [L,H,W]
```

### Why This Is Correct

1. **Beer-Lambert Law**: Standard physics model for opacity/transmission
2. **Monotonically increasing**: Opacity never decreases
3. **Correct boundaries**:
   - opacity(0) = 0
   - opacity(∞) → 1.0
4. **Smooth gradients everywhere**: Good for optimization
5. **Physically meaningful**: More thickness = more opacity

### Test Results (After Fix)

```
thick_ratio=0.0: opacity=0.0000  ✓ Correct
thick_ratio=0.1: opacity=0.2212  ✓ Reasonable
thick_ratio=0.5: opacity=0.7135  ✓ Increasing
thick_ratio=1.0: opacity=0.9179  ✓ High opacity
thick_ratio=2.0: opacity=0.9933  ✓ Near saturation
thick_ratio=3.0: opacity=0.9994  ✓ Fully saturated
```

**Decreasing intervals**: 0 (correct!)

---

## Implementation

### Files Modified

1. **[src/autoforge/Helper/OptimizerHelper.py](../src/autoforge/Helper/OptimizerHelper.py)**
   - `composite_image_cont()`: Lines 176-183
   - `composite_image_disc()`: Lines 298-305

### Changes

Both functions now use:

```python
k_opacity = 2.5
opac = 1.0 - torch.exp(-k_opacity * thick_ratio)
```

Instead of the old 4-parameter polynomial-log formula.

---

## Testing

### Tests Created

1. **[test_bug15_opacity_formula.py](test_bug15_opacity_formula.py)**
   - Confirms the original bug
   - Shows the decreasing behavior
   - Plots the problem

2. **[test_bug15_fix_proposal.py](test_bug15_fix_proposal.py)**
   - Compares different k values
   - Shows why k=2.5 is optimal
   - Generates comparison plots

3. **[test_bug15_fix_verification.py](test_bug15_fix_verification.py)**
   - 7 comprehensive tests
   - Verifies monotonicity
   - Checks boundary values
   - Tests gradient flow
   - Validates composite functions
   - Tests with realistic TD values
   - Checks extreme cases

### Test Results

```
✓ test_opacity_monotonic           - 0 decreasing intervals
✓ test_opacity_bounds               - Correct at 0, 1, and ∞
✓ test_opacity_gradients            - All positive and smooth
✓ test_composite_cont_smoke         - Works correctly
✓ test_composite_disc_smoke         - Works correctly  
✓ test_opacity_with_various_tds     - Handles real TD values
✓ test_no_negative_opacity_at_extremes - No numerical issues
```

**All 7 tests PASSED**

### Regression Testing

Ran existing test suite to ensure no breakage:

```
✓ test_optimizer_smoke.py           - 2/2 passed
✓ test_loss_functions.py            - 2/2 passed
✓ test_pruning_helper.py            - 3/3 passed
✓ test_filament_helper.py           - 4/4 passed
✓ test_image_helper.py              - 4/4 passed
✓ test_output_helper.py             - 4/4 passed
✓ test_bug10_fix_verification.py    - 2/2 passed
✓ test_bug11_fixed_validation.py    - 3/3 passed
✓ test_bug13_complete_verification.py - 2/2 passed
✓ test_bug14_fix_verification.py    - 1/1 passed
```

**27 tests PASSED, 0 failed**

---

## Impact

### Positive Changes

1. **Physically correct** opacity behavior
2. **Better optimization** due to smooth gradients everywhere
3. **More realistic** layer compositing
4. **No more** opacity decreasing with thickness
5. **Predictable** behavior: thicker = more opaque

### Potential Side Effects

- Output images may look **slightly different** due to changed opacity curves
- Users may need to **re-tune** their layer heights/counts for optimal results
- The change is **incompatible** with projects optimized using the old formula

### Recommendations

- Users should **re-optimize** existing projects with the new formula
- Document the change in release notes
- Consider adding a **legacy mode** flag if backward compatibility is critical

---

## Calibration of k=2.5

The calibration constant `k=2.5` was chosen to provide:

- **Opacity ≈ 92%** at thick_ratio = 1.0 (when thickness equals TD)
- **Reasonable gradient** for optimization
- **Good saturation** behavior

### Alternative Values

- **k=2.0**: More gradual saturation (opacity ≈ 86% at ratio=1.0)
- **k=3.0**: Faster saturation (opacity ≈ 95% at ratio=1.0)

If empirical testing shows different behavior is desired, adjust `k_opacity` constant.

---

## Physics Background

**Transmissivity Distance (TD)**: The thickness at which a material achieves a certain opacity level.

**Beer-Lambert Law**:

```
I_transmitted = I_incident * exp(-α * d)
opacity = 1 - I_transmitted / I_incident = 1 - exp(-α * d)
```

Where:

- `α` is the absorption coefficient
- `d` is the thickness
- For our formula: `thick_ratio = d / TD`, and `k = α * TD`

This is the **standard physics model** for how light passes through semi-transparent materials.

---

## Conclusion

Bug #15 is **completely fixed**. The opacity formula now:

✅ Follows correct physics (Beer-Lambert law)  
✅ Monotonically increases with thickness  
✅ Has smooth gradients for optimization  
✅ Handles all edge cases correctly  
✅ Passes all tests  
✅ No regressions in existing functionality  

The fix improves both the **physical correctness** and **optimization quality** of the layer compositing system.
