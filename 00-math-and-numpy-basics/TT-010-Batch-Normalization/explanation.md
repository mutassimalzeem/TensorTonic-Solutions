# Explanation for TT-010 — Batch Normalization Forward (NumPy)

Batch Norm normalizes each feature across batch: zero-mean unit-variance, then affine transform.

## Steps in `solution.py`:

1. Convert `x, gamma, beta` to float np.arrays.
2. **4D case** (N,C,H,W): 
   - `mean = np.mean(x, axis=(0,2,3), keepdims=True)` – per channel
   - `var = np.var(x, axis=(0,2,3), keepdims=True)`
   - Reshape `gamma/beta` to (1,C,1,1) broadcast
3. **2D case**: `mean/var axis=0` (per feature).
4. `x_norm = (x - mean) / np.sqrt(var + eps)` – standardize
5. `y = gamma * x_norm + beta` – scale/shift

## Notes on behavior
- Supports image (4D) and tabular (2D) data.
- `keepdims=True` for broadcasting.
- eps prevents div0.
- Output shape matches input.

## Output shape
Same as x.

