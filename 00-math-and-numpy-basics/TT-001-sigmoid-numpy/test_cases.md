# Test Cases for TT-001 — Sigmoid (NumPy)

## 1. Scalar input
- Input: `0`
- Expected output: `0.5`

## 2. Simple list input
- Input: `[0, 1, -1]`
- Expected output: `array([0.5, 0.73105858, 0.26894142])`

## 3. Large positive and negative values
- Input: `np.array([-1000, 0, 1000])`
- Expected output: `array([0.0, 0.5, 1.0])`

## 4. Multi-dimensional input
- Input:
  ```python
  np.array([[0, 2], [-2, 5]])
  ```
- Expected output:
  ```python
  array([[0.5, 0.88079708], [0.11920292, 0.99330715]])
  ```

## 5. Shape preservation
- Input: `np.zeros((2, 3))`
- Expected output: an array of shape `(2, 3)` filled with `0.5`
