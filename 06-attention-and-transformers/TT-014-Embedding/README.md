# TT-014 — Embedding

## Problem Link
TensorTonic: Embedding

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement a basic **embedding layer** using PyTorch and a helper function that converts token IDs into scaled embedding vectors.

The scaling step multiplies the embeddings by `sqrt(d_model)`, which is a common choice in transformer-style architectures.

---

## Official Problem Statement
Create a PyTorch embedding layer for a given vocabulary size and model dimension, then use it to embed token indices and scale the output by `sqrt(d_model)`.

---

## Constraints
- `vocab_size` is the number of unique token IDs
- `d_model` is the embedding dimension
- The embedding layer is created with `nn.Embedding`
- Input tokens must be a PyTorch tensor of integer token IDs
- The embedded output is scaled by `math.sqrt(d_model)`

---

## Example

### Input
```python
embedding = create_embedding_layer(vocab_size=10, d_model=4)
tokens = torch.tensor([1, 3, 5])
output = embed_tokens(embedding, tokens, d_model=4)
```

### Output
```python
torch.Tensor of shape (3, 4)
```

---

## Solution Overview

The implementation in `solution.py`:

- creates an embedding layer with `nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)`
- looks up token vectors by calling the embedding layer on the input tensor
- scales the resulting embeddings with `math.sqrt(d_model)`

This keeps the implementation short and aligned with the standard embedding scaling used in transformer models.

---

## Usage

```python
import torch
from solution import create_embedding_layer, embed_tokens

embedding = create_embedding_layer(vocab_size=10, d_model=4)
tokens = torch.tensor([1, 3, 5])

output = embed_tokens(embedding, tokens, d_model=4)
print(output.shape)
```
