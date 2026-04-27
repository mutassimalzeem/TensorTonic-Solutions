# Explanation for TT-016 — Scaled Dot-Product Attention (PyTorch)

Scaled dot-product attention is the core mechanism in Transformers.

## Formula

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

## Steps in `solution.py`:

1. `d_k = Q.shape[-1]` — get key dimension
2. `scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)` — compute similarity scores with scaling
3. `weights = F.softmax(scores, dim=-1)` — normalize to probabilities
4. `return weights @ V` — weighted sum of values

## Why scaling?

- Prevents dot products from growing too large with high d_k
- Keeps softmax gradients stable
- `sqrt(d_k)` is the standard scaling factor

## Notes on behavior

- Supports batched inputs (batch, seq, d_k)
- `transpose(-2, -1)` swaps last two dims for matrix multiplication
- Softmax applied over keys (dim=-1)
- Output shape matches V: (batch, seq_len_Q, d_v)

## Output shape

(batch, seq_len_Q, d_v) where d_v = V.shape[-1]
