# Notes for TT-012 — Euclidean Distance (NumPy)

- L2 distance: geometric straight-line between points.
- Vectorized `(x - y)**2` → sum → sqrt.
- Shape check prevents invalid compute.
- Print debug on mismatch (not raise/error).
- Handles scalars (1,) or arrays.
- Edges: identical vectors d=0; 1D; high-dim; zero vec.
- O(N) time N=dim.
- Alt: `np.linalg.norm(x-y, ord=2)`.
- Used in kNN, loss, embedding similarity.


