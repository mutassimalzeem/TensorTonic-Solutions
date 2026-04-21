# Notes for TT-008 — Dot Product (NumPy)

- Dot product: Σ x[i]*y[i], commutative (x·y = y·x).
- `np.array(x/y)` converts lists to arrays automatically.
- Length check prevents shape mismatch error.
- `x * y` elementwise multiply (Hadamard), then `np.sum` aggregates.
- `np.float64(...)` ensures float output as required.
- 1D vectors only (per docstring); 2D would need matrix multiply.
- Edge cases: zero-length? (len=0 ok, sum=0.0), scalars as 1-element.
- Vectorized vs loop: np faster for large n.
- Alternative: `np.dot(x, y)` but impl uses sum for teaching.
- Time: O(n), Space: O(n) temp arrays.

