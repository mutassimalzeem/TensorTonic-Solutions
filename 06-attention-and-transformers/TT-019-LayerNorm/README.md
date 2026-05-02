# TT-019 — Layer Normalization (NumPy)

## Problem Link
TensorTonic: Layer Normalization (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **Layer Normalization** as used in Transformers.

Normalizes activations across features (last dimension) for each sample independently:
```
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

---

## Official Problem Statement
Implement layer normalization that normalizes across the last feature dimension.

---

## Constraints
- `x`: input array of any shape
- `gamma, beta`: learnable scale/shift parameters matching last dim
- `axis = -1` (normalize over last dimension)
- `eps = 1e-6` for numerical stability
- Output same shape as input

---

## Example

### Input
```python
x = [[1., 2., 3., 4.]]
gamma = [1., 1., 1., 1.]
beta = [0., 0., 0., 0.]
```

### Output
```python
# Mean ~2.5, Var ~1.67, normalized to zero mean, unit var
```

---

## Solution Overview

In `solution.py`:
```python
mean = np.mean(x, axis=-1, keepdims=True)
var = np.var(x, axis=-1, keepdims=True)
norm = (x - mean) / np.sqrt(var + eps)
return gamma * norm + beta
```

Standard LayerNorm formula.

---

## Usage

```python
import numpy as np
from solution import layer_norm

x = np.random.randn(2, 4, 8)
gamma = np.ones(8)
beta = np.zeros(8)
output = layer_norm(x, gamma, beta)
print(output.shape)  # (2, 4, 8)
