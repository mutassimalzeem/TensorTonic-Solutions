# Test Cases for TT-009 — Linear Regression Closed Form (NumPy)

## 1. Simple line y = 2x
- Input:
  ```python
  X = np.array([[1, 1], [1, 2], [1, 3]])
  y = np.array([2, 4, 6])
  ```
- Expected output:
  ```python
  array([0., 2.])  # w0=0, w1=2
  ```

## 2. With intercept y = x + 1
- Input:
  ```python
  X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
  y = np.array([1, 2, 3, 4])
  ```
- Expected output:
  ```python
  array([1., 1.])
  ```

## 3. Single feature minimal
- Input:
  ```python
  X = np.array([[1, 1]])
  y = np.array([3])
  ```
- Expected output:
  ```python
  array([3.])
  ```

## 4. Noisy data approx fit
- Input:
  ```python
  X = np.array([[1, 1], [1, 2], [1, 4]])
  y = np.array([1, 2, 4.5])
  ```
- Expected output ≈:
  ```python
  array([0.15, 1.05])
  ```

## 5. Perfect fit multiple features
- Input (2 features):
  ```python
  X = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9]])
  y = np.array([2, 5, 14])  # y = x1 + x2
  ```
- Expected output:
  ```python
  array([0., 1., 1.])
  ```

