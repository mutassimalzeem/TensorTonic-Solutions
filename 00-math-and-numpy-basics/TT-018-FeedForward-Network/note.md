# Notes for TT-018 — FeedForward Network (NumPy)

- Position-wise MLP applied independently to each token.
- Two linear transformations with ReLU nonlinearity between.
- Typically `d_ff = 4 × d_model` (expansion ratio).
- Second layer projects back to `d_model` (bottleneck).
- Uses `np.dot()` = `@` matrix multiplication.
- `np.maximum(0, hidden)` is element-wise ReLU.
- Bias `b1`, `b2` broadcast across `batch × seq`.
- No residual connection or layer norm (separate Transformer components).
- No dropout (training feature).
- In full Transformer: `output = LayerNorm(x + FFN(x))`.
- NumPy broadcasting handles multi-dimensional inputs automatically.
- Forward-only implementation (no backward pass).
- Common `d_model=512`, `d_ff=2048`.
- PyTorch equivalent: `torch.nn.Sequential(Linear(d_model, d_ff), ReLU(), Linear(d_ff, d_model))`.
- Applied after attention sublayer in encoder/decoder layers.


