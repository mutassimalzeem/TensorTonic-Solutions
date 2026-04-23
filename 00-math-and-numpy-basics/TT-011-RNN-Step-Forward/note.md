# Notes for TT-011 — RNN Step Forward (NumPy)

- Single timestep RNN cell (vanilla RNN).
- Input flexibility: scalars/lists/arrays → ensured 1D.
- Scalar param=0 → auto zero-init matrix/vector.
- `@` matmul efficient.
- tanh nonlinear activation [-1,1].
- No caching/updates (forward only).
- Extends to seq by loop over t.
- Edges: D=1/H=1, zero x_t/h_prev, zero Wh.
- Time O(D H + H^2), Space O(H).
- Test with identity Wh → h_t ≈ tanh(h_prev + proj x).


