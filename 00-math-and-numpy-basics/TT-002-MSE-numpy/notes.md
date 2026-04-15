# Notes for TT-002 — Mean Squared Error (NumPy)

- The function converts inputs to NumPy arrays before performing any numerical operations.
- Shape compatibility is checked using `np.shape(y_pred) == np.shape(y_true)`.
- If the prediction and target arrays have different shapes, the implementation returns `None` instead of raising an error.
- The implementation uses `np.square` and `np.mean` to compute the mean squared error.
- The solution is designed for simplicity and correctness while following NumPy best practices.
