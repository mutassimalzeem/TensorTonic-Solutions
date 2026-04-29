# Explanation for TT-018 — FeedForward Network (NumPy)

The feed-forward network is a position-wise fully connected layer applied independently to each token embedding in Transformer layers.

## Architecture

```
input: x ∈ ℝ^(batch × seq × d_model)
↓
hidden = x @ W1 + b1 ∈ ℝ^(batch × seq × d_ff)
↓ ReLU
relu = max(0, hidden)
↓
output = relu @ W2 + b2 ∈ ℝ^(batch × seq × d_model)
```

Typical `d_ff = 4 × d_model` (e.g., 2048 vs 512).

## Steps in `solution.py`:

### 1. First Linear Layer
```python
hidden1 = np.dot(x, W1) + b1
```
- `x @ W1`: `(batch, seq, d_model) @ (d_model, d_ff)` → `(batch, seq, d_ff)`
- `+ b1`: broadcast bias across batch/seq

### 2. ReLU Activation
```python
relu = np.maximum(0, hidden1)
```
- Element-wise: negative → 0, positive unchanged
- Non-linear transformation expands representational power

### 3. Second Linear Layer
```python
hidden2 = np.dot(relu, W2) + b2
```
- `relu @ W2`: `(batch, seq, d_ff) @ (d_ff, d_model)` → `(batch, seq, d_model)`
- `+ b2`: final bias, projects back to model dimension

## Notes on behavior

- **Position-wise**: Same weights applied to every token independently
- **Expansion**: Typically `d_ff > d_model` (bottleneck in second layer)
- **Residual**: Full Transformer layer: `LayerNorm(x + FFN(x))`
- **Dropout**: Training only (not implemented here)
- **Shape preservation**: output shape matches input
- **Broadcasting**: Biases automatically broadcast across batch/seq dims

## Usage Context

FFN follows multi-head attention in each Transformer layer:
```
LayerNorm → MultiHeadAttention → Add → LayerNorm → FFN → Add
```

## Complexity

- Time: O(batch × seq × (d_model × d_ff + d_ff × d_model))
- Space: O(batch × seq × d_ff) — hidden layer


