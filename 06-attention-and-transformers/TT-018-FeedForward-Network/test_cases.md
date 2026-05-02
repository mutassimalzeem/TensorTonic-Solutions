# Test Cases for TT-018 — FeedForward Network (NumPy)

## 1. Basic forward pass
- Input:
  ```python
  batch_size, seq_len, d_model, d_ff = 2, 3, 4, 8
  x = np.array([[[1., 2., 3., 4.]]])  # (1,1,4)
  W1 = np.eye(4, 8) * 0.5
  b1 = np.zeros(8)
  W2 = np.eye(8, 4) * 0.5
  b2 = np.zeros(4)
  ```
- Expected: ReLU(non-negative) → linear projection, shape `(1, 1, 4)`

## 2. ReLU activation test
- Input:
  ```python
  x = np.array([[[1., -1., 0.]]])  # mixed signs
  W1 = np.ones((1, 2)) * 0.5
  b1 = np.array([0., 0.])
  W2 = np.ones((2, 1))
  b2 = np.array([0.])
  ```
- Expected: negative input → 0 after ReLU, shape preserved

## 3. Batch and sequence dimensions
- Input:
  ```python
  x = np.random.randn(2, 4, 8)
  W1 = np.random.randn(8, 16) * 0.1
  b1 = np.zeros(16)
  W2 = np.random.randn(16, 8) * 0.1
  b2 = np.zeros(8)
  ```
- Expected output shape: `(2, 4, 8)`

## 4. All-negative input (ReLU zeros)
- Input:
  ```python
  x = np.ones((1, 1, 4)) * -1
  W1 = np.eye(4, 8)
  b1 = np.zeros(8) * -2  # ensure hidden negative
  W2 = np.eye(8, 4)
  b2 = np.zeros(4)
  ```
- Expected: output all zeros (ReLU kills signal)

## 5. Realistic dimensions
- Input:
  ```python
  batch_size, seq_len = 4, 64
  d_model, d_ff = 512, 2048
  x = np.random.randn(batch_size, seq_len, d_model)
  W1 = np.random.randn(d_model, d_ff) * 0.01
  b1 = np.zeros(d_ff)
  W2 = np.random.randn(d_ff, d_model) * 0.01
  b2 = np.zeros(d_model)
  ```
- Expected output shape: `(4, 64, 512)`


