# Explanation for TT-012 — Euclidean Distance (NumPy)

Euclidean distance (L2 norm diff):
```
d = sqrt( sum((x - y)^2 ) )
```

## Steps in `solution.py`:

1. `x = np.array(x)`, `y = np.array(y)` convert inputs.
2. `if x.shape == y.shape:` check compatibility.
3. `z = np.sum((x - y)**2)` squared L2.
4. `return np.sqrt(z)` root → distance.
5. Else: `print("Dead", x.shape, y.shape)` error msg.

## Notes
- Vectorized elementwise.
- Works scalars/1D/multi-dim same shape.
- No return on mismatch (debug print).
- Equivalent `np.linalg.norm(x - y)` but manual.

## Output
Scalar float.



