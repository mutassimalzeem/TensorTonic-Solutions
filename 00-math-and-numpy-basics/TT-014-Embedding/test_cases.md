# Test Cases for TT-014 — Embedding

## 1. Create embedding layer
- Input:
  ```python
  embedding = create_embedding_layer(vocab_size=10, d_model=4)
  ```
- Expected output: an `nn.Embedding` layer with `10` embeddings of dimension `4`

## 2. Embed a 1D token tensor
- Input:
  ```python
  embedding = create_embedding_layer(vocab_size=10, d_model=4)
  tokens = torch.tensor([1, 3, 5])
  output = embed_tokens(embedding, tokens, d_model=4)
  ```
- Expected output: a tensor of shape `(3, 4)`

## 3. Embed a 2D token tensor
- Input:
  ```python
  embedding = create_embedding_layer(vocab_size=20, d_model=8)
  tokens = torch.tensor([[1, 2], [3, 4]])
  output = embed_tokens(embedding, tokens, d_model=8)
  ```
- Expected output: a tensor of shape `(2, 2, 8)`

## 4. Scaling check
- Input:
  ```python
  embedding = create_embedding_layer(vocab_size=5, d_model=4)
  tokens = torch.tensor([0])
  raw = embedding(tokens)
  scaled = embed_tokens(embedding, tokens, d_model=4)
  ```
- Expected output: `scaled` should be equal to `raw * 2.0`

## 5. Invalid token index
- Input:
  ```python
  embedding = create_embedding_layer(vocab_size=5, d_model=4)
  tokens = torch.tensor([7])
  output = embed_tokens(embedding, tokens, d_model=4)
  ```
- Expected output: PyTorch should raise an index error because token `7` is outside the vocabulary range
