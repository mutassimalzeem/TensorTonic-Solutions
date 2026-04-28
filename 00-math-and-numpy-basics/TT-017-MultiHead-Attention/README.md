# TT-017 — Multi-Head Attention (NumPy)

## Problem Link
TensorTonic: Multi-Head Attention (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **multi-head attention** from scratch using NumPy.

Multi-head attention runs scaled dot-product attention in parallel across `h` heads, allowing the model to attend to information from different representation subspaces.

---

## Official Problem Statement
Implement multi-head attention with linear projections, head splitting, parallel attention, head concatenation, and output projection.

---

## Constraints
- Q, K, V: np.ndarray shape `(batch_size, seq_len, d_model)`
- W_q, W_k, W_v, W_o: learned weight matrices
- `num_heads`: number of parallel attention heads
- `d_model` must be divisible by `num_heads`
- Each head operates with `d_k = d_model // num_heads`
- Use manual `softmax` implementation (not scipy)

---

## Example

### Input
```python
batch_size, seq_len, d_model = 2, 4, 8
num_heads = 2

Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)
```

### Output
```python
output = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
# shape: (2, 4, 8)
```

---

## Solution Overview

In `solution.py`:

1. **Linear projections**: `Q = Q @ W_q`, `K = K @ W_k`, `V = V @ W_v`
2. **Split heads**: Reshape from `(batch, seq, d_model)` to `(batch, num_heads, seq, d_k)`
3. **Attention scores**: `scores = Q @ K.transpose(...)`
4. **Scale**: `scores / np.sqrt(d_k)`
5. **Softmax**: Manual stable softmax per head
6. **Weighted sum**: `attention_weights @ V`
7. **Combine heads**: Transpose and reshape back to `(batch, seq, d_model)`
8. **Output projection**: `output = attention @ W_o`

---

## Usage

```python
import numpy as np
from solution import multi_head_attention

batch_size, seq_len, d_model = 2, 4, 8
num_heads = 2

Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_o = np.random.randn(d_model, d_model)

output = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
print(output.shape)  # (2, 4, 8)
```

