# TT-020 - Encoder Block (NumPy)

## Problem Link
TensorTonic: Encoder Block (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement a complete **Transformer encoder block** using NumPy.

An encoder block combines:

1. Multi-head self-attention
2. Residual connection and layer normalization
3. Position-wise feed-forward network
4. Second residual connection and layer normalization

The block keeps the same input and output shape:

```python
(batch_size, seq_len, d_model)
```

---

## Official Problem Statement
Implement a Transformer encoder block with multi-head attention, feed-forward network, residual connections, and layer normalization.

---

## Constraints
- `x`: input tensor with shape `(batch_size, seq_len, d_model)`
- `W_q, W_k, W_v, W_o`: attention projection matrices
- `W1, b1, W2, b2`: feed-forward network parameters
- `gamma1, beta1`: LayerNorm parameters after attention
- `gamma2, beta2`: LayerNorm parameters after feed-forward
- `num_heads`: number of attention heads
- `d_model` must be divisible by `num_heads`
- Output shape must match `x`

---

## Example

### Input
```python
batch_size, seq_len, d_model, d_ff = 2, 4, 8, 16
num_heads = 2

x = np.random.randn(batch_size, seq_len, d_model)

W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1

W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)

gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)
gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)
```

### Output
```python
output = encoder_block(
    x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
    gamma1, beta1, gamma2, beta2, num_heads
)
# shape: (2, 4, 8)
```

---

## Solution Overview

In `solution.py`:

```python
mha_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
first_layer = layer_norm(x + mha_output, gamma1, beta1)
ffn_output = feed_forward(first_layer, W1, b1, W2, b2)
output = layer_norm(first_layer + ffn_output, gamma2, beta2)
```

This matches the standard post-norm Transformer encoder pattern:

```text
x -> SelfAttention -> Add & Norm -> FeedForward -> Add & Norm
```

---

## Usage

```python
import numpy as np
from solution import encoder_block

batch_size, seq_len, d_model, d_ff = 2, 4, 8, 16
num_heads = 2

x = np.random.randn(batch_size, seq_len, d_model)
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1
W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)
gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)
gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)

output = encoder_block(
    x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
    gamma1, beta1, gamma2, beta2, num_heads
)

print(output.shape)  # (2, 4, 8)
```
