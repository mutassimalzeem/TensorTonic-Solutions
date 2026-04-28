# Notes for TT-017 — Multi-Head Attention (NumPy)

- Multi-head attention = multiple parallel scaled dot-product attentions.
- `d_model` must be divisible by `num_heads` for clean splitting.
- Each head sees `d_k = d_model // num_heads` dimensions.
- Head splitting via `reshape + transpose` is more efficient than loops.
- `transpose(0, 2, 1, 3)` moves head dimension to batch-like position for parallel matmul.
- Manual `softmax` with stability trick: `exp(x - max(x))` prevents overflow.
- No masking implemented (no causal or padding mask).
- No dropout on attention weights (training detail).
- Weight matrices `W_q, W_k, W_v, W_o` are typically learned parameters.
- Self-attention: Q = K = V from same source.
- Cross-attention: Q from decoder, K/V from encoder.
- PyTorch equivalent: `torch.nn.MultiheadAttention` (optimized, fused).
- Attention pattern: `(batch, heads, seq, seq)` — each token attends to all tokens.
- Output projection `W_o` mixes information across heads before feeding forward.

