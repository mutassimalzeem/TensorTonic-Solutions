# Test Cases for TT-003 — Softmax

## 1. Scalar input
- Input: `2.0`
- Expected output: `array([1.0])`

## 2. Simple list input
- Input: `[1.0, 2.0, 3.0]`
- Expected output: `array([0.09003057, 0.24472847, 0.66524096])`

## 3. Negative and positive values
- Input: `np.array([-1.0, 0.0, 1.0])`
- Expected output: `array([0.09003057, 0.24472847, 0.66524096])`

## 4. 2D array input (rowwise probability normalization)
- Input:
  ```python
  np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
  ```
- Expected output:
  ```python
  array([
      [0.09003057, 0.24472847, 0.66524096],
      [0.33333333, 0.33333333, 0.33333333]
  ])
  ```

## 5. Shape preservation
- Input: `np.zeros((2, 3))`
- Expected output: an array of shape `(2, 3)` where each row is `[0.33333333, 0.33333333, 0.33333333]`

## Notes
- For multi-dimensional arrays, normalization is always applied along the last axis.
- All output rows or slices along the last axis should sum to 1.
