# TT-008 — Dot Product (NumPy)

## Problem Link
TensorTonic: Dot Product (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement the **dot product** (scalar product) of two 1D vectors x and y of the same length.

The dot product is defined as:

```
dot(x, y) = Σ x[i] * y[i] = x · y
```

---

## Official Problem Statement
Implement vectorized dot product using NumPy that accepts lists or arrays, returns float, handles 1D only, raises ValueError for mismatched lengths.

---

## Constraints
- Inputs: scalars, lists, or 1D NumPy arrays of same length
- Output: single float64 scalar
- Vectorized (no Python loops)
- Raise ValueError if lengths differ
- 1D vectors only (per requirements)

---

## Example

### Input
```python
x = [1, 2, 3]
y = [4, 5, 6]
```

### Output
```python
32.0
```

(1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32)

---

## Solution Overview

The implementation in `solution.py`:

- Converts `x = np.array(x)`, `y = np.array(y)`
- Checks `if len(x) != len(y): raise ValueError(...)`
- Returns `np.float64(np.sum(x * y))` – elementwise multiply, sum, cast to float64

Vectorized and efficient.

---

## Usage

```python
from solution import dot_product

result = dot_product([1, 2, 3], [4, 5, 6])
print(result)  # 32.0
```

