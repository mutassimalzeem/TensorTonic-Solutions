# Explanation for TT-011 — RNN Step Forward (NumPy)

RNN step computes hidden state update:

```
pre_act = x_t · Wx + h_{t-1} · Wh + b
h_t = tanh(pre_act)
```

## Steps in `solution.py`:

1. `np.atleast_1d(*).astype(float)` ensure vectors.
2. Detect dims from x_t.shape[0]=D, h_prev.shape[0]=H.
3. Handle scalar 0 params → zero matrices/vectors.
4. Matrix mults: `x_t @ Wx` (D->H), `h_prev @ Wh` (H->H).
5. `preact = ... + b` (broadcast).
6. `np.tanh(preact)` elementwise.

## Notes
- Vectorized matmul + tanh.
- Defaults allow easy zero-init testing.
- Shape out: (H,)
- Robust scalar/list inputs.

## Output shape
(Dx1 @ D x H) → H, always (H,).


