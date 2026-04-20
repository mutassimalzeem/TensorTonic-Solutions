# Test Cases for TT-007 — Matrix Transposition (NumPy)

## 1. 2x3 rectangular matrix
- Input:
  ```python
  np.array([[1, 2, 3], [4, 5, 6]])
  ```
- Expected output:
  ```python
  array([[1, 4],
         [2, 5],
         [3, 6]])
  ```

## 2. 2x2 square matrix
- Input:
  ```python
  np.array([[1, 2], [3, 4]])
  ```
- Expected output:
  ```python
  array([[1, 3],
         [2, 4]])
  ```

## 3. 1x1 scalar matrix
- Input:
  ```python
  np.array([[5]])
  ```
- Expected output:
  ```python
  array([[5]])
  ```

## 4. 1x3 row vector
- Input:
  ```python
  np.array([[1, 2, 3]])
  ```
- Expected output:
  ```python
  array([[1],
         [2],
         [3]])
  ```

## 5. 3x1 column vector (as 3x1 matrix)
- Input:
  ```python
  np.array([[1], [2], [3]])
  ```
- Expected output:
  ```python
  array([[1, 2, 3]])
