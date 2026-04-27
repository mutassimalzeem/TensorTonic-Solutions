# Test Cases for TT-016 — Scaled Dot-Product Attention (PyTorch)

## 1. Identity attention
- Input:
  ```python
  Q = torch.tensor([[[1., 0.], [0., 1.]]])
  K = torch.tensor([[[1., 0.], [0., 1.]]])
  V = torch.tensor([[[1., 2.], [3., 4.]]])
  ```
- Expected:
  ```python
  tensor([[[1., 2.],
           [3., 4.]]])
  ```

## 2. Uniform Query attends equally
- Input:
  ```python
  Q = torch.ones(1, 2, 4)
  K = torch.eye(4).unsqueeze(0).expand(1, 4, 4)
  V = torch.arange(16).float().reshape(1, 4, 4)
  ```
- Expected: weighted average of all V rows (approx equal weights)

## 3. Batch of 2 sequences
- Input:
  ```python
  Q = torch.randn(2, 3, 8)
  K = torch.randn(2, 3, 8)
  V = torch.randn(2, 3, 8)
  ```
- Expected output shape: `(2, 3, 8)`

## 4. Scaling effect (d_k=1 vs d_k=100)
- Input:
  ```python
  Q = torch.ones(1, 1, 100) * 10
  K = torch.ones(1, 1, 100) * 10
  V = torch.ones(1, 1, 100)
  ```
- Expected: scores = 10000/sqrt(100) = 1000, softmax ≈ 1, output ≈ V

## 5. Cross-attention different seq lengths
- Input:
  ```python
  Q = torch.randn(1, 2, 4)
  K = torch.randn(1, 5, 4)
  V = torch.randn(1, 5, 6)
  ```
- Expected output shape: `(1, 2, 6)`
