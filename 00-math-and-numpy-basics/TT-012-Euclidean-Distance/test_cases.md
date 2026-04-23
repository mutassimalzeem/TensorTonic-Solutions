# Test Cases for TT-012 — Euclidean Distance (NumPy)

## 1. Basic vectors
- Input:
  ```python
  x = [1.0, 2.0]
  y = [4.0, 6.0]
  ```
- Expected:
  ```python
  5.0
  ```

## 2. Identical
- Input:
  x = [0,0], y = [0,0]
- Expected:
  ```python
  0.0
  ```

## 3. Scalar
- Input:
  x = 3.0, y = 0.0
- Expected:
  ```python
  3.0
  ```

## 4. 3D
- Input:
  x = np.array([1,2,3]), y = np.array([4,5,6])
- Expected:
  ```python
  5.196152422706632
  ```

## 5. Shape mismatch (debug)
- Input:
  x = [1,2], y = [1]
- Expected print:
  ```
  Dead (2,) (1,)
  ```


