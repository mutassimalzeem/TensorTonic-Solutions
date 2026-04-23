# Test Cases for TT-010 — Batch Normalization Forward (NumPy)

## 1. Simple 2D batch
- Input:
  ```python
  x = np.array([[1., 2.], [3., 4.]])
  gamma = np.ones(2)
  beta = np.zeros(2)
  ```
- Expected output ≈:
  ```python
  array([[-1., -1.],
         [ 1.,  1.]])
  ```

## 2. 4D single channel image batch
- Input:
  ```python
  x = np.ones((2, 1, 2, 2)) * np.array([1., 2.])[:,None,None,None]
  gamma = np.array([1.])
  beta = np.array([0.])
  ```
- Expected ≈ normalized zero-mean unit-var

## 3. Scale/shift effect
- Input:
  ```python
  x = np.array([[0., 10.]])
  gamma = np.array([2., 0.5])
  beta = np.array([1., 3.])
  ```
- Expected:
  ```python
  array([[5., -1.]])  # after norm then affine
  ```

## 4. Batch size 1
- Input:
  ```python
  x = np.array([[5., 6.]])
  gamma = np.ones(2)
  beta = np.zeros(2)
  ```
- Expected:
  ```python
  array([[0., 0.]])  # var=0, eps norm
  ```

## 5. Multi-channel 4D
- Input (N=1,C=2,H=W=1):
  ```python
  x = np.array([[[[1., 2.]]]])
  gamma, beta = np.ones(2), np.zeros(2)
  ```
- Expected zero mean per channel.

