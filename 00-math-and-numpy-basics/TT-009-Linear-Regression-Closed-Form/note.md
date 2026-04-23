# Notes for TT-009 — Linear Regression Closed Form (NumPy)

- Normal equation: w = (X^T X)^(-1) X^T y – exact OLS solution.
- `np.linalg.inv` for square matrix invert.
- X must have intercept (column of 1s).
- Fails if X^T X singular (collinear features).
- No regularization (ridge would add λI).
- `np.array(X), np.array(y)` calls present but ineffective (void).
- Edge cases: n=d+1 minimal data, perfect fit.
- Use `np.linalg.lstsq` alternative for stability/large n.
- Time: O(min(n,d)^3) approx, Space: O(n d).
- Assumes X shape (n_samples, n_features+1).

