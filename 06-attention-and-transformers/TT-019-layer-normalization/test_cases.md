# Test Cases for TT-019 — Layer Normalization (NumPy)

## 1. Simple 1D array (identity gamma/beta)
- Input:
  ```python
  x = [[1., 2., 3., 4.]]
  gamma = [1., 1., 1., 1.]
  beta = [0., 0., 0., 0.]
  ```
- Expected: normalized to zero mean, unit variance

## 2. Scaled/shifted output
- Input:
  ```python
  x = [[1., 2., 3., 4.]]
  gamma = [2., 2., 2., 2.]
  beta = [1., 1., 1., 1.]
  ```
- Expected: normalized × 2 + 1

## 3. 2D batch (each row normalized independently)
- Input:
  ```python
  x = [[1., 2., 3.], [4., 5., 6.]]
  gamma = [1., 1., 1.]
  beta = [0., 0., 0.]
  ```
- Expected: each row normalized separately, both zero mean

## 4. 3D input (batch, seq, features)
- Input:
  ```python
  x = np.random.randn(2, 4, 8)
  gamma = np.ones(8)
  beta = np.zeros(8)
  ```
- Expected output shape: `(2, 4, 8)`

## 5. Very small values (numerical stability)
- Input:
  ```python
  x = [[1e-8, 2e-8, 3e-8]]
  gamma = [1., 1., 1.]
  beta = [0., 0., 0.]
  ```
- Expected: finite output (eps prevents inf)
