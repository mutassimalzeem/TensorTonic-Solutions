# Notes for TT-020 - Encoder Block (NumPy)

- Encoder block = self-attention + feed-forward network.
- Uses post-norm structure: sublayer output is added to input, then normalized.
- Self-attention uses `Q = K = V = x`.
- Multi-head attention lets tokens attend to other positions in parallel subspaces.
- Residual connection 1: `x + MHA(x)`.
- LayerNorm 1 stabilizes the attention sublayer output.
- Feed-forward network is position-wise, so every token passes through the same MLP.
- Residual connection 2: `first_layer + FFN(first_layer)`.
- LayerNorm 2 stabilizes the final encoder output.
- Input and output shapes stay `(batch_size, seq_len, d_model)`.
- `gamma1`, `beta1`, `gamma2`, and `beta2` are learnable LayerNorm parameters.
- `d_model` should be divisible by `num_heads`.
- This implementation is forward-only and does not include dropout or masking.
- A full Transformer encoder stacks this block multiple times.
- In practice, `d_ff` is often around `4 x d_model`.
