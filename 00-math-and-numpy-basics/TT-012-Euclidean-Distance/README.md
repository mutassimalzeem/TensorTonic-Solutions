# TT-012 — Euclidean Distance (NumPy)

## Problem Link
TensorTonic: Euclidean Distance (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **Euclidean (L2) distance** between two vectors: sqrt( Σ (x_i - y_i)^2 )

||x - y||_2

---

## Official Problem Statement
Compute L2 distance assuming same shape vectors. Print shapes if mismatch.

---

## Constraints
- x, y: scalars/lists/arrays same shape
- Output: float scalar
- Shape mismatch: print "Dead" + shapes

---

## Example

### Input
```python
x = np.array([1.0, 2.0])
y = np.array([4.0, 6.0])
```

### Output
```python
5.0
```

---

## Solution Overview

In `solution.py`:
- `x = np.array(x)`, `y = np.array(y)`
- If same shape:
  `np.sqrt(np.sum((x - y)**2))`
- Else: print("Dead", x.shape, y.shape)

Vectorized diff-square-sum-sqrt.

---

## Usage

```python
import numpy as np
from solution import euclidean_distance

dist = euclidean_distance([1,2], [4,6])
print(dist)  # 5.0
```


