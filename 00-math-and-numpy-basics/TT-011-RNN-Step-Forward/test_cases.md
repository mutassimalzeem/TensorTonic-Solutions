# Test Cases for TT-011 — RNN Step Forward (NumPy)

## 1. Basic step (from code)
- Input:
  ```python
  x_t = [0.0, 0.0], h_prev = [1.0, -1.0], Wx=0, Wh=np.eye(2), b=0
  ```
- Expected ≈
  ```python
  array([0.76159416, 0.        ])
  ```

## 2. Non-zero Wx
- Input:
  ```python
  x_t = [1.0, 0.0], h_prev = [0.0, 0.0], Wx=np.eye(2), Wh=0, b=0
  ```
- Expected ≈
  ```python
  array([0.76159416, 0.        ])
  ```

## 3. Scalar zero defaults
- Input:
  x_t=[1], h_prev=[0], Wx=0, Wh=0, b=0
- Expected ≈
  ```python
  array([0.])
  ```

## 4. Full interaction
- Input:
  x_t=[1,2], h_prev=[0.5,-0.5], Wx=[[1,0],[0,1]], Wh=[[0,1],[1,0]], b=[0,0]
- Expected: tanh([1*0.5 + 2* (-0.5) + cross terms]) compute numerically.

## 5. 1D minimal
- Input:
  x_t=[0], h_prev=[1], Wx=0, Wh=1, b=0
- Expected:
  ```python
  array([tanh(1)])
  ```


