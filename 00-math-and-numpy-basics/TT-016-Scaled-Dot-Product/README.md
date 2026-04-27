# TT-016 — Scaled Dot-Product Attention (PyTorch)

## Problem Link
TensorTonic: Scaled Dot-Product Attention (PyTorch)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **scaled dot-product attention** as used in Transformers.

Formula:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

---

## Official Problem Statement
Compute attention weights from Query, Key, Value tensors using scaled dot-product.

---

## Constraints
- Q, K, V: torch.Tensor (batch, seq_len, d_k) or compatible
- d_k: last dim of Q
- Use `math.sqrt(d_k)` for scaling
- `F.softmax(scores, dim=-1)` for weights
- Output: weighted sum of V

---

## Example

### Input
```python
Q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
K = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
V = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
```

### Output
```python
tensor([[[1.0, 2.0],
         [3.0, 4.0]]])
```

---

## Solution Overview

In `solution.py`:
- `d_k = Q.shape[-1]`
- `scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)`
- `weights = F.softmax(scores, dim=-1)`
- `return weights @ V`

Standard Transformer attention.

---

## Usage

```python
import torch
from solution import scaled_dot_product_attention

Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)
out = scaled_dot_product_attention(Q, K, V)
print(out.shape)  # (2, 4, 8)
