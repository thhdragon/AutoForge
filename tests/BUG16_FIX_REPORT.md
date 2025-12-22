# Bug 16 Fix Report: No Bounds Check in Bleed Layer Effect

**Date**: December 22, 2025  
**Status**: ✅ FIXED  
**Severity**: 🟠 HIGH  
**Category**: Bounds Checking

---

## Summary

Fixed a critical bounds violation in the `bleed_layer_effect()` function where the output could exceed the valid opacity range [0,1], causing rendering artifacts.

---

## Bug Description

**Location**: [OptimizerHelper.py](../src/autoforge/Helper/OptimizerHelper.py#L131)

The `bleed_layer_effect()` function applies edge bleeding by adding `strength * blurred` to the original mask. When:

- `strength` is high (e.g., 0.5 or greater)
- Input mask values are near 1.0
- Neighboring pixels are also high

The output can exceed 1.0, violating the valid opacity range.

**Original Code**:

```python
def bleed_layer_effect(mask: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    # ... kernel setup ...
    blurred = F.conv2d(mask.unsqueeze(1), kernel, padding=1, groups=1).squeeze(1)
    return mask + strength * blurred  # ❌ No bounds check!
```

**Impact**:

- Invalid opacity values > 1.0
- Rendering artifacts in final output
- Unrealistic layer blending
- Potential numerical instability

---

## Verification

Created comprehensive test suite in [test_bug16_bleed_bounds.py](test_bug16_bleed_bounds.py).

### Test Results (Before Fix)

1. **Uniform High Mask Test**: ❌ FAILED
   - Input: 5×5 mask of 0.9 values, strength=0.5
   - Expected (unclamped): 1.35
   - Actual output range: [1.0687, 1.3500]
   - **Bug confirmed**: Values exceeded 1.0

2. **High Strength Test**: ✅ Passed (edge case)
   - Checkerboard pattern [0, 1], strength=2.0
   - Output: [0.5, 1.0]

3. **3D Multi-layer Test**: ❌ FAILED
   - Input: 3×4×4 tensor of 0.95 values, strength=0.4
   - Expected (unclamped): 1.33
   - Actual output range: [1.0925, 1.3300]
   - **Bug confirmed**: Values exceeded 1.0

4. **Realistic Scenario Test**: ❌ FAILED
   - Sigmoid-activated random mask, strength=0.1
   - **69 out of 100 pixels violated bounds**
   - Output range: [0.8196, 1.0872]
   - **Bug confirmed**: Common in real optimization scenarios

---

## Fix Implementation

**Modified Code**:

```python
def bleed_layer_effect(mask: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    # ... kernel setup ...
    blurred = F.conv2d(mask.unsqueeze(1), kernel, padding=1, groups=1).squeeze(1)
    
    # Combine original mask with bleed from neighbors
    # Clamp to [0,1] to prevent invalid opacity values (Bug #16 fix)
    return torch.clamp(mask + strength * blurred, 0.0, 1.0)
```

**Change**: Added `torch.clamp(..., 0.0, 1.0)` to ensure output stays in valid range.

---

## Verification (After Fix)

All tests now pass:

1. **Uniform High Mask Test**: ✅ PASSED
   - Input would produce 1.35 unclamped
   - Output correctly clamped: [1.0000, 1.0000]

2. **High Strength Test**: ✅ PASSED
   - Output: [0.5000, 1.0000] ✓

3. **3D Multi-layer Test**: ✅ PASSED
   - Input would produce 1.33 unclamped
   - Output correctly clamped: [1.0000, 1.0000]

4. **Realistic Scenario Test**: ✅ PASSED
   - **0 out of 100 pixels violated bounds**
   - Output range: [0.6378, 1.0000] ✓

---

## Performance Impact

- **Minimal**: `torch.clamp()` is a fast element-wise operation
- No change to algorithm logic
- JIT compilation still works (function decorated with `@torch.jit.script`)

---

## Related Code

The `bleed_layer_effect()` function is called in two places:

1. [Line 172](../src/autoforge/Helper/OptimizerHelper.py#L172): `composite_image_cont()`

   ```python
   p_print_bleed = bleed_layer_effect(p_print, strength=0.1)
   ```

2. [Line 298](../src/autoforge/Helper/OptimizerHelper.py#L298): `composite_image_disc()`

   ```python
   p_print_bleed = bleed_layer_effect(p_print, strength=0.1)
   ```

Both now benefit from proper bounds checking.

---

## Lessons Learned

1. **Always validate bounds** when combining tensors with coefficients
2. **Test extreme values** (high strength, high mask values)
3. **Realistic scenarios** often expose edge cases (69% of pixels violated!)
4. **Opacity/probability values** should always be in [0,1]

---

## Files Modified

- ✅ [src/autoforge/Helper/OptimizerHelper.py](../src/autoforge/Helper/OptimizerHelper.py#L131) - Added clamp
- ✅ [tests/test_bug16_bleed_bounds.py](test_bug16_bleed_bounds.py) - Created verification tests

---

## Conclusion

Bug 16 has been **successfully fixed and verified**. The bleed layer effect now properly clamps output values to the valid [0,1] range, preventing rendering artifacts and ensuring numerical stability.

**Status**: ✅ RESOLVED
