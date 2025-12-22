# Bug #9 Fix: Negative Tau Schedule Prevention

## Summary

Fixed critical bug in tau annealing schedule that could prevent model discretization if `init_tau < final_tau`.

## The Bug

In [Optimizer.py](src/autoforge/Modules/Optimizer.py#L156), the tau decay rate was calculated as:

```python
self.decay_rate = (self.init_tau - self.final_tau) / (args.iterations - self.warmup_steps)
```

If a user accidentally swapped the arguments or misconfigured them such that `init_tau < final_tau`, the `decay_rate` would become **negative**. This means tau would **increase** over time instead of cooling down (annealing), preventing the Gumbel-Softmax from discretizing the solution.

### Impact

- Temperature never decreases → model stays in soft continuous mode
- Discrete solution never forms → output quality degrades
- Optimization becomes ineffective

## The Fix

### 1. Added Parameter Validation (lines 91-96)

```python
# Validate tau schedule parameters
if self.init_tau < self.final_tau:
    raise ValueError(
        f"init_tau ({self.init_tau}) must be >= final_tau ({self.final_tau}). "
        f"Tau annealing requires init_tau >= final_tau for temperature to cool over time."
    )
```

### 2. Protected Against Division Issues (lines 160-162)

```python
# Compute decay rate with protection against division by near-zero denominator
iterations_after_warmup = max(1, args.iterations - self.warmup_steps)
self.decay_rate = (self.init_tau - self.final_tau) / iterations_after_warmup
```

Added `max(1, ...)` to prevent potential division by zero if `args.iterations <= warmup_steps`.

## Verification

### Test Added

Added comprehensive test `test_bug9_tau_schedule_validation()` in [tests/test_optimizer_smoke.py](tests/test_optimizer_smoke.py) that:

1. **Validates rejection**: Confirms that `init_tau < final_tau` raises `ValueError` with clear error message
2. **Validates acceptance**: Confirms that `init_tau >= final_tau` works correctly
3. **Checks constraint**: Asserts that `optimizer.init_tau >= optimizer.final_tau` after construction

### Test Results

```
tests/test_optimizer_smoke.py::test_bug9_tau_schedule_validation PASSED [100%]
tests/test_optimizer_smoke.py::test_optimizer_one_step_cpu PASSED [100%]
```

## Files Modified

- [src/autoforge/Modules/Optimizer.py](src/autoforge/Modules/Optimizer.py)
  - Added validation at lines 91-96
  - Updated decay_rate calculation at lines 160-162
- [tests/test_optimizer_smoke.py](tests/test_optimizer_smoke.py)
  - Added test function `test_bug9_tau_schedule_validation()`

## Backward Compatibility

✅ **Fully backward compatible**

- Valid configurations (where `init_tau >= final_tau`) work exactly as before
- Only rejects invalid configurations that would have caused silent failures
- Provides clear error messages for misconfiguration
