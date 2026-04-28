# Test Cases for TT-017 — Multi-Head Attention (NumPy)

## 1. Basic forward pass
- Input:
  ```python
  batch_size, seq_len, d_model = 2, 4, 8
  num_heads = 2
  Q = np.ones((batch_size, seq_len, d_model))
  K = np.ones((batch_size, seq_len, d_model))
  V = np.ones((batch_size, seq_len, d_model))
  W_q = W_k = W_v = W_o = np.eye(d_model)
  ```
- Expected output shape: `(2, 4, 8)`
- Attention weights should be uniform since Q=K

## 2. Identity weights, different V
- Input:
  ```python
  Q = K = V = np.arange(24).reshape(1, 3, 8).astype(float)
  W_q = W_k = W_v = W_o = np.eye(8)
  num_heads = 2
  ```
- Expected: output shape `(1, 3, 8)`, attention pattern reflects token similarities

## 3. Single head (num_heads=1)
- Input:
  ```python
  d_model = 4
  num_heads = 1
  Q = K = V = np.random.randn(1, 2, 4)
  W_q = W_k = W_v = W_o = np.eye(4)
  ```
- Expected: equivalent to single scaled dot-product attention, shape `(1, 2, 4)`

## 4. Different Q, K, V (cross-attention setup)
- Input:
  ```python
  Q = np.random.randn(1, 2, 8)
  K = np.random.randn(1, 5, 8)
  V = np.random.randn(1, 5, 8)
  W_q = W_k = W_v = W_o = np.eye(8)
  num_heads = 2
  ```
- Expected output shape: `(1, 2, 8)` — Q seq_len preserved

## 5. d_model=64, num_heads=8 (realistic)
- Input:
  ```python
  batch_size, seq_len = 4, 16
  d_model, num_heads = 64, 8
  Q = K = V = np.random.randn(batch_size, seq_len, d_model)
  W_q = np.random.randn(d_model, d_model) * 0.01
  W_k = np.random.randn(d_model, d_model) * 0.01
  W_v = np.random.randn(d_model, d_model) * 0.01
  W_o = np.random.randn(d_model, d_model) * 0.01
  ```
- Expected output shape: `(4, 16, 64)`
- Attention weights shape internally: `(4, 8, 16, 16)`

