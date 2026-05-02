# TT-010 — Batch Normalization Forward (NumPy)

## Problem Link
TensorTonic: Batch Normalization Forward (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **batch normalization forward pass** for training: normalize per feature, scale/shift with gamma/beta.

Formula:
```
x_norm = (x - μ) / √(σ² + ε)
y = γ x_norm + β
```

---

## Official Problem Statement
Implement batch_norm_forward for 4D (N,C,H,W image) or 2D data, mean/var over batch/spatial dims.

---

## Constraints
- x: float array 4D or 2D
- gamma, beta: 1D per-channel params (C,)
- eps=1e-5 numerical stability
- Normalize over (0,2,3) for 4D, axis=0 for 2D
- Output same shape as x

---

## Example

### Input (2D)
```python
x = np.array([[1., 2.], [3., 4.]])
gamma = np.ones(2)
beta = np.zeros(2)
```

### Output ≈
```python
array([[-1., -1.],
       [ 1.,  1.]])
```

---

## Solution Overview

The implementation in `solution.py`:

- Convert to float arrays
- If 4D: mean/var axis=(0,2,3) keepdims, reshape gamma/beta (1,C,1,1)
- Else: mean/var axis=0
- `x_norm = (x - mean) / np.sqrt(var + eps)`
- `y = gamma * x_norm + beta`

Handles CNN/data formats.

---

## Usage

```python
import numpy as np
from solution import batch_norm_forward

x = np.array([[1., 2.], [3., 4.]])
gamma, beta = np.ones(2), np.zeros(2)
out = batch_norm_forward(x, gamma, beta)
print(out)
```

