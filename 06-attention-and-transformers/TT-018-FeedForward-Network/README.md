# TT-018 — FeedForward Network (NumPy)

## Problem Link
TensorTonic: FeedForward Network (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **position-wise feed-forward network** as used in Transformer layers.

Two linear layers with ReLU activation between them:
```
FFN(x) = W2 · ReLU(W1 · x + b1) + b2
```

Applied independently to each position (token embedding).

---

## Official Problem Statement
Implement the position-wise feed-forward sub-layer with residual connection and layer norm typically found after multi-head attention in Transformers.

---

## Constraints
- `x`: input shape `(batch_size, seq_len, d_model)`
- `W1, b1`: first layer weights/bias (d_model → d_ff)
- `W2, b2`: second layer weights/bias (d_ff → d_model)
- Use `np.maximum(0, hidden)` for ReLU
- Output same shape as input

---

## Example

### Input
```python
batch_size, seq_len, d_model, d_ff = 2, 4, 8, 16
x = np.random.randn(batch_size, seq_len, d_model)
W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)
```

### Output
```python
output = feed_forward(x, W1, b1, W2, b2)
# shape: (2, 4, 8)
```

---

## Solution Overview

In `solution.py`:
```
hidden1 = x @ W1 + b1
relu = np.maximum(0, hidden1)
output = relu @ W2 + b2
```

Simple two-layer MLP applied position-wise.

---

## Usage

```python
import numpy as np
from solution import feed_forward

batch_size, seq_len, d_model = 2, 4, 8
d_ff = 16

x = np.random.randn(batch_size, seq_len, d_model)
W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)

output = feed_forward(x, W1, b1, W2, b2)
print(output.shape)  # (2, 4, 8)
```

