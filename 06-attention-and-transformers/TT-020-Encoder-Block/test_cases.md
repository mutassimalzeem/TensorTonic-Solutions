# Test Cases for TT-020 - Encoder Block (NumPy)

## 1. Basic forward pass
- Input:
  ```python
  batch_size, seq_len, d_model, d_ff = 2, 4, 8, 16
  num_heads = 2
  x = np.random.randn(batch_size, seq_len, d_model)
  ```
- Expected output shape: `(2, 4, 8)`
- Expected output values: finite numbers

## 2. Identity-style attention projections
- Input:
  ```python
  batch_size, seq_len, d_model, d_ff = 1, 3, 4, 8
  num_heads = 2
  x = np.arange(12).reshape(1, 3, 4).astype(float)
  W_q = W_k = W_v = W_o = np.eye(d_model)
  ```
- Expected: output shape `(1, 3, 4)`
- Attention should be deterministic because all projection weights are identity matrices.

## 3. LayerNorm identity parameters
- Input:
  ```python
  gamma1 = np.ones(d_model)
  beta1 = np.zeros(d_model)
  gamma2 = np.ones(d_model)
  beta2 = np.zeros(d_model)
  ```
- Expected: LayerNorm normalizes each token feature vector without extra scaling or shifting.

## 4. Feed-forward expansion
- Input:
  ```python
  d_model = 8
  d_ff = 32
  W1 = np.random.randn(d_model, d_ff) * 0.1
  b1 = np.zeros(d_ff)
  W2 = np.random.randn(d_ff, d_model) * 0.1
  b2 = np.zeros(d_model)
  ```
- Expected: feed-forward hidden dimension expands to `d_ff`, then returns to `d_model`.

## 5. Single attention head
- Input:
  ```python
  batch_size, seq_len, d_model, d_ff = 1, 2, 4, 8
  num_heads = 1
  ```
- Expected output shape: `(1, 2, 4)`
- Multi-head attention behaves like one scaled dot-product attention head.

## 6. Numerical stability
- Input:
  ```python
  x = np.full((1, 2, 4), 1e-8)
  gamma1 = gamma2 = np.ones(4)
  beta1 = beta2 = np.zeros(4)
  ```
- Expected: output contains finite values because LayerNorm uses `eps=1e-6`.
