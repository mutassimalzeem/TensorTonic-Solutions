# Notes for TT-016 — Scaled Dot-Product Attention (PyTorch)

- Core Transformer attention mechanism (Vaswani et al. 2017).
- Q, K, V typically projections from same input via learned W_q, W_k, W_v.
- Scaling by sqrt(d_k) prevents softmax saturation.
- `transpose(-2, -1)` generalizes to any batch dims.
- `F.softmax` stable with dim=-1 over keys.
- Complexity: O(n^2 * d) for seq_len n.
- Self-attention: Q=K=V from same source.
- Cross-attention: Q from decoder, K/V from encoder.
- Masking (not impl): causal/padding masks before softmax.
- Multi-head: parallel h heads with d_k = d_model / h.
- PyTorch native: `torch.nn.functional.scaled_dot_product_attention` (fused).
