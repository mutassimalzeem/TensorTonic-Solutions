# Explanation for TT-009 — Linear Regression Closed Form (NumPy)

Closed-form solution minimizes squared error: w = argmin ||Xw - y||^2 = (X^T X)^(-1) X^T y

## Steps in `solution.py`:

1. `X^T X`: `np.dot(np.transpose(X), X)` – Gram matrix.
2. Invert: `np.linalg.inv(...)` – assumes invertible.
3. `X^T y`: `np.dot(np.transpose(X), y)` – correlation vector.
4. Multiply: `np.dot(w1, w2)` – optimal weights.

Note: `np.array(X/y)` calls discarded (no effect).

## Notes on behavior
- O(d^3) from invert (d=features+1).
- Requires full rank X^T X.
- X should have bias column (1s).
- Output shape: (d+1,).

## Output shape
Input X (n,d+1), y (n,) → w (d+1,)

