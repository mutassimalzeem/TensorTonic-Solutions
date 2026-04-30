# Explanation for TT-019 — Layer Normalization (NumPy)

Layer normalization normalizes activations across features (last axis) independently for each sample.

## Formula

```
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

Where:
- mean = E[x] over last axis
- var = Var[x] over last axis

## Steps in `solution.py`:

### 1. Compute Mean
```python
mean = np.mean(x, axis=-1, keepdims=True)
```
- Averages over last dimension (features)
- keepdims=True keeps shape for broadcasting

### 2. Compute Variance
```python
var = np.var(x, axis=-1, keepdims=True)
```
- Variance = E[(x - mean)²]
- NumPy uses unbiased var by default (ddof=0)

### 3. Normalize
```python
norm = (x - mean) / np.sqrt(var + eps)
```
- Center: subtract mean
- Scale: divide by sqrt(variance)
- eps prevents division by zero

### 4. Learnable Parameters
```python
output = gamma * norm + beta
```
- gamma: scaling factor (learnable)
- beta: shifting factor (learnable)
- Both broadcast over samples

## Notes on behavior

- **Sample-independent**: Each sample normalized separately (unlike BatchNorm)
- **Last axis**: Works for any input shape (2D, 3D, etc.)
- **No mini-batch**: Unlike BatchNorm, LayerNorm doesn't depend on batch
- **Trainable**: gamma starts at 1, beta at 0 (or custom)
- **Stable**: eps=1e-6 prevents numerical issues
- **In Transformer**: Applied before and after attention/FFN

## Difference: LayerNorm vs BatchNorm

| | LayerNorm | BatchNorm |
|---|---|---|
| Axis | Feature (-1) | Batch (0) |
| Training depends on | None | Mini-batch |
| Inference | Same | Fixed stats |

## Complexity

Time: O(batch × seq × d_model)
Space: O(batch × seq) for mean/var
