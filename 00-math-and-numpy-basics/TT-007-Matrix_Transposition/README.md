# TT-007 — Matrix Transposition (NumPy)

## Problem Link
TensorTonic: Matrix Transposition (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **matrix transposition** using NumPy that converts a matrix A (m × n) to its transpose B (n × m).

The transpose operation swaps rows and columns such that B[j][i] = A[i][j] for all i, j.

---

## Official Problem Statement
Implement matrix transposition manually using nested loops, without using NumPy's `transpose()` or `.T` attribute.

---

## Constraints
- Input is a 2D NumPy array (rectangular matrix)
- Output must be a NumPy array with transposed shape (n × m)
- Use manual loops to build transpose (teaching purpose)
- Preserve input data type
- The implementation gets dimensions with `np.shape(A)`

---

## Example

### Input
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
```

### Output
```python
array([[1, 4],
       [2, 5],
       [3, 6]])
```

---

## Solution Overview

The implementation in `solution.py`:

- Gets `rows = np.shape(A)[0]`, `cols = np.shape(A)[1]`
- Creates zero-initialized list-of-lists `B` of size `cols x rows`
- Double loop: `B[j][i] = A[i][j]` for i in rows, j in cols
- Returns `np.array(B)`

This is O(m*n) time, fully manual transpose.

---

## Usage

```python
import numpy as np
from solution import matrix_transpose

A = np.array([[1, 2, 3], [4, 5, 6]])
result = matrix_transpose(A)
print(result)
