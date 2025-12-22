# AutoForge Remaining Bugs

**Last Updated**: December 22, 2025  
**Status**: 16 unsolved bugs remaining (7 high, 9 medium)  
**Solved Bugs**: See [bugs.done.md](bugs.done.md) - 18 bugs fixed

---

## Executive Summary

Critical path is now clear. These 16 remaining bugs are organized by severity and category. Bugs #10, #16, #20, and #22-35 remain to be fixed. **Bugs #11, #14, and #18 have been fixed.**

---

## 🔴 CRITICAL BUGS - REMAINING

### 10. Material Index Out-of-Bounds Access

**File**: [OutputHelper.py](src/autoforge/Helper/OutputHelper.py#L260-L275)  
**Severity**: 🔴 CRITICAL  
**Category**: Bounds

```python
for idx in filament_indices:
    mat = material_data[idx]  # No bounds check!
```

**Issue**: `filament_indices` from `extract_filament_swaps()` not validated. Could exceed `len(material_data)`.  
**Impact**: KeyError crash when accessing invalid material index.  
**Fix**:

```python
for idx in filament_indices:
    if not (0 <= idx < len(material_data)):
        raise ValueError(f"Invalid material index {idx}, have {len(material_data)} materials")
    mat = material_data[idx]
```

---

## 🟠 HIGH PRIORITY BUGS

### 16. No Bounds Check in Bleed Layer Effect

**File**: [OptimizerHelper.py](src/autoforge/Helper/OptimizerHelper.py#L250-L260)  
**Severity**: 🟠 HIGH  
**Category**: Bounds

```python
blurred = F.conv2d(mask.unsqueeze(1), kernel, padding=1, groups=1)
return mask + strength * blurred  # No clamp!
```

**Issue**: Output can exceed [0,1] if strength is high or blurred is large.  
**Impact**: Invalid opacity values > 1.0; rendering artifacts.  
**Fix**:

```python
return torch.clamp(mask + strength * blurred, 0.0, 1.0)
```

---

### 20. CSV Hex Color Parsing No Validation

**File**: [FilamentHelper.py](src/autoforge/Helper/FilamentHelper.py#L110-L120)  
**Severity**: 🟠 HIGH  
**Category**: Validation

```python
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return [int(hex_str[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
```

**Issue**: No format validation. Invalid hex like `#ZZZZZZ` crashes with ValueError. Also doesn't handle 3-char hex `#ABC`.  
**Impact**: Poor error messages; program crashes on bad CSV.  
**Fix**:

```python
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c*2 for c in hex_str])  # Expand #ABC to #AABBCC
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    try:
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
        return [r/255.0, g/255.0, b/255.0]
    except ValueError:
        raise ValueError(f"Invalid hex digits in {hex_str}")
```

---

## 🟡 MEDIUM PRIORITY BUGS

### 22. Empty Cluster Handling in KMeans Splitting

**File**: [DepthEstimateHeightMap.py](src/autoforge/Init_comparer/DepthEstimateHeightMap.py#L160-L180)  
**Severity**: 🟡 MEDIUM  
**Category**: Logic

```python
for split in range(k_split):
    sub_mask = split_labels == split
    inds = indices[sub_mask]
    if inds.size == 0:
        continue  # Skip but cluster_id still incremented later?
    new_cluster_id += 1
```

**Issue**: Empty sub-clusters are skipped but cluster ID numbering gets misaligned.  
**Impact**: Cluster ID gaps; some layer assignments invalid.  
**Fix**:

```python
for split in range(k_split):
    sub_mask = split_labels == split
    inds = indices[sub_mask]
    if inds.size == 0:
        continue
    # Only increment if cluster created
    new_cluster_id += 1
```

---

### 23. Image Resize Aspect Ratio Distortion

**File**: [ImageHelper.py](src/autoforge/Helper/ImageHelper.py#L30-L40)  
**Severity**: 🟡 MEDIUM  
**Category**: Input

```python
new_w = int(round(w_img * scale))
new_h = int(round(h_img * scale))
```

**Issue**: Independent rounding breaks aspect ratio. Example: 100x101 at scale 0.5 → 50x50 (squeezed!).  
**Impact**: Input image distorted; affects color matching quality.  
**Fix**:

```python
new_w = int(round(w_img * scale))
new_h = int(h_img * scale)  # Don't round second dimension
```

---

### 24. Silent Dominant Color Computation Failure

**File**: [auto_forge.py](src/autoforge/auto_forge.py#L420-L440)  
**Severity**: 🟡 MEDIUM  
**Category**: Error Handling

```python
res = _compute_dominant_image_color(img_rgb, alpha)
if res is not None:
    dominant_hex, dominant_rgb = res
else:
    print("Warning: Auto background color computation failed...")
    # No traceback of actual error!
```

**Issue**: Exceptions caught internally with `except Exception`, silently return None. No debugging info.  
**Impact**: Hard to debug when auto background fails.  
**Fix**:

```python
try:
    res = _compute_dominant_image_color(img_rgb, alpha)
except Exception as e:
    import traceback
    print(f"Warning: Auto background color failed: {e}")
    traceback.print_exc()
    res = None
```

---

### 25. KMeans Over-Clustering on Small Images

**File**: [DepthEstimateHeightMap.py](src/autoforge/Init_comparer/DepthEstimateHeightMap.py#L220-L240)  
**Severity**: 🟡 MEDIUM  
**Category**: Init

```python
optimal_n = max_layers  # Default to max_layers
labels = KMeans(n_clusters=optimal_n, random_state=seed).fit_predict(pixels)
```

**Issue**: If image is small (e.g., 10x10 = 100 pixels) and max_layers=75, KMeans creates 75 clusters from 100 pixels. Most clusters have 1-2 pixels.  
**Impact**: Useless height initialization; clustering noise.  
**Fix**:

```python
max_clusters = min(max_layers, len(np.unique(pixels)))
optimal_n = max(2, min(max_layers, max_clusters))
```

---

### 26. Swaps Extraction Appends Last Material Twice

**File**: [OutputHelper.py](src/autoforge/Helper/OutputHelper.py#L240-L260)  
**Severity**: 🟡 MEDIUM  
**Category**: Logic

```python
for i in range(1, L):
    current = int(disc_global[i])
    if current != prev:
        filament_indices.append(current)
        slider_values.append(i + 1)
        prev = current

filament_indices.append(prev)  # Append AGAIN!
```

**Issue**: Last material appended twice, creating redundant filament in output.  
**Impact**: Confusing swap instructions; extra material listed.  
**Fix**:

```python
# Check if different from last before appending
if filament_indices[-1] != prev:
    filament_indices.append(prev)
```

---

### 27. Early Stopping Condition Unreachable

**File**: [auto_forge.py](src/autoforge/auto_forge.py#L950-L960)  
**Severity**: 🟡 MEDIUM  
**Category**: Optimization

```python
if (
    optimizer.best_step is not None
    and optimizer.num_steps_done - optimizer.best_step > args.early_stopping
):
    break
```

**Issue**: `best_step` stays None until first discrete improvement. Early stopping only triggers after that, making it almost useless.  
**Impact**: Wastes iterations; doesn't early-stop on continuous loss plateau.  
**Fix**: Track improvement in continuous loss too:

```python
continuous_no_improve = self.num_steps_done - self.best_continuous_step
if continuous_no_improve > args.early_stopping:
    break
```

---

### 28. TensorBoard Writer Not Closed

**File**: [Optimizer.py](src/autoforge/Modules/Optimizer.py#L180-L195)  
**Severity**: 🟡 MEDIUM  
**Category**: Resource

```python
if args.tensorboard:
    self.writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
# Never closed!

def __del__(self):
    if self.writer is not None:
        self.writer.close()  # __del__ not guaranteed
```

**Issue**: `__del__` not guaranteed to run immediately. File handle leak.  
**Impact**: Open file handles; TensorBoard directory locked.  
**Fix**:

```python
def close(self):
    if self.writer is not None:
        self.writer.close()
        self.writer = None

# Call at end of optimization:
try:
    # ... optimization ...
finally:
    optimizer.close()
```

---

### 29. Multi-Run Failure Handling Crashes

**File**: [auto_forge.py](src/autoforge/auto_forge.py#L1090-L1110)  
**Severity**: 🟡 MEDIUM  
**Category**: Error Handling

```python
ret = []
for run in range(args.num_runs):
    try:
        loss = start(...)
        ret.append((run, loss))
    except Exception:
        traceback.print_exc()

best_run = min(ret, key=lambda x: x[1])  # Crashes if ret is empty!
```

**Issue**: If all runs fail, `ret` is empty and `min(ret)` raises ValueError.  
**Impact**: Crashes instead of graceful error message.  
**Fix**:

```python
if not ret:
    raise RuntimeError(f"All {args.num_runs} runs failed!")
best_run = min(ret, key=lambda x: x[1])
```

---

### 30. Bleed Mask Kernel Excludes Center Pixel

**File**: [OptimizerHelper.py](src/autoforge/Helper/OptimizerHelper.py#L240-L250)  
**Severity**: 🟡 MEDIUM  
**Category**: Logic

```python
kernel = [[1,1,1], [1,0,1], [1,1,1]] / 8.0  # Center is 0!
blurred = F.conv2d(mask.unsqueeze(1), kernel, padding=1, groups=1)
```

**Issue**: Kernel takes average of neighbors only, ignoring the pixel itself. Weak bleed effect.  
**Impact**: Edge softening is weak; layer boundaries remain sharp.  
**Fix**:

```python
kernel = [[1,1,1], [1,1,1], [1,1,1]] / 9.0  # Include center
```

---

### 31. Type Hints Missing Across Modules

**File**: [PruningHelper.py](src/autoforge/Helper/PruningHelper.py), [OptimizerHelper.py](src/autoforge/Helper/OptimizerHelper.py)  
**Severity**: 🟡 MEDIUM  
**Category**: Code Quality

```python
def _chunked(iterable, chunk_size):  # Missing types!
    ...

def compose_image_continuous(...):  # Many missing types
    ...
```

**Issue**: Incomplete type hints reduce IDE support and refactoring safety.  
**Impact**: Harder to maintain; more runtime errors.  
**Fix**: Add full type hints:

```python
from typing import Iterator, List
def _chunked(iterable: List, chunk_size: int) -> Iterator:
    ...
```

---

### 32. Input CSV Structure Not Validated

**File**: [FilamentHelper.py](src/autoforge/Helper/FilamentHelper.py#L75-L90)  
**Severity**: 🟡 MEDIUM  
**Category**: Validation

```python
df = load_materials_pandas(args)
df["Brand"]  # Assumes column exists!
```

**Issue**: No check for required columns: "Brand", "Name", "Color", "Transmissivity".  
**Impact**: Cryptic KeyError if user provides wrong CSV format.  
**Fix**:

```python
required_cols = {"Brand", "Name", "Color", "Transmissivity"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")
```

---

### 34. Depth Map Clustering Over-Fits to Noise

**File**: [DepthEstimateHeightMap.py](src/autoforge/Init_comparer/DepthEstimateHeightMap.py#L250-L270)  
**Severity**: 🟡 MEDIUM  
**Category**: Init

```python
labels = KMeans(n_clusters=optimal_n, ...).fit_predict(pixels)
```

**Issue**: No denoising. Noisy depth maps create spurious clusters.  
**Impact**: Poor height initialization; clustering artifacts.  
**Fix**: Add bilateral filtering:

```python
import cv2
depth_smooth = cv2.bilateralFilter(depth_map, 9, 75, 75)
```

---

### 35. Silent Silhouette Score Errors

**File**: [ChristofidesHeightMap.py](src/autoforge/Init_comparer/ChristofidesHeightMap.py#L120-L130)  
**Severity**: 🟡 MEDIUM  
**Category**: Code Quality

```python
try:
    return silhouette_score(X_subset, lbl_subset, metric="euclidean")
except ValueError:
    return -1.0  # Sentinel value, never checked!
```

**Issue**: Returns -1.0 on error but caller doesn't check for sentinel.  
**Impact**: Bad clustering selected as "best"; poor height initialization.  
**Fix**:

```python
try:
    score = silhouette_score(X_subset, lbl_subset, metric="euclidean")
    if np.isnan(score):
        return -1.0
    return score
except ValueError as e:
    print(f"Warning: Silhouette score failed: {e}")
    return -1.0

# In caller:
if best_score <= -0.5:  # Check for bad score
    print("Warning: Clustering quality poor")
```

---

## 📊 Remaining Bugs Summary

| Category | Count | Bugs |
|----------|-------|------|
| Bounds | 2 | 10, 16 |
| Validation | 2 | 20, 32 |
| Logic | 3 | 22, 26, 30 |
| Init | 3 | 18, 25, 34 |
| Error Handling | 2 | 24, 29 |
| Code Quality | 2 | 31, 35 |
| Input | 1 | 23 |
| Resource | 1 | 28 |
| Optimization | 1 | 27 |
| **Total** | **17** | **10, 16, 18, 20, 22-35 (excluding #14 - solved)** |

---

## 🎯 Recommended Fix Order

1. **Critical Path** (Bug #10): ✅ Bug #11 FIXED, ✅ Bug #14 FIXED - Bounds checking remaining
2. **High Priority** (Bugs #16, #18, #20): Numerical and initialization
3. **Medium Priority** (Bugs #22-35): Code quality and robustness

---

**Last Updated**: December 22, 2025  
**Codebase**: AutoForge (3D print path optimizer)
2 | 20, 32 |
| Logic | 4 | 11, 22, 26, 30 |
| Init | 4 | 14, 18, 25, 34 |
| Numerical | 1 | 14 |
| Error Handling | 2 | 24, 29 |
| Code Quality | 2 | 31, 35 |
| Input | 1 | 23 |
| Resource | 1 | 28 |
| Optimization | 1 | 27 |
| Schedule | 1 | 9 |
| **Total** | **20** | **99-11): Schedule
