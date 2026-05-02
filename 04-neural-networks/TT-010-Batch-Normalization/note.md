# Notes for TT-010 — Batch Normalization Forward (NumPy)

- Normalizes per feature/channel across batch (dims 0+spatial).
- 4D: axis=(0,2,3) for N,C,H,W → mean/var (1,C,1,1)
- 2D: axis=0 → mean/var (C,)
- gamma/beta reshape for broadcast.
- `np.var` population variance (ddof=0 default).
- eps=1e-5 standard for stability.
- During inference: use running mean/var.
- Benefits: faster training, less sensitive to init.
- Edge: batch_size=1 var=0 (eps saves).
- Time: O(batch * spatial * C), Space: O(1) extra.
- Extends to backward pass (not impl).

