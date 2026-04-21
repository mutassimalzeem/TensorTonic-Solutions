# Test Cases for TT-008 — Dot Product (NumPy)

## 1. Basic vectors
- Input:
  ```python
  x = [1, 2, 3]
  y = [4, 5, 6]
  ```
- Expected output:
  ```python
  32.0
  ```

## 2. Zero vectors
- Input:
  ```python
  x = [0, 0]
  y = [1, 2]
  ```
- Expected output:
  ```python
  0.0
  ```

## 3. Single element
- Input:
  ```python
  x = [5]
  y = [2]
  ```
- Expected output:
  ```python
  10.0
  ```

## 4. Negative values
- Input:
  ```python
  x = [-1, 1]
  y = [1, -1]
  ```
- Expected output:
  ```python
  -2.0
  ```

## 5. Mismatched lengths (error)
- Input:
  ```python
  x = [1, 2]
  y = [3, 4, 5]
  ```
- Expected: `ValueError: Vectors must be of the same length`

