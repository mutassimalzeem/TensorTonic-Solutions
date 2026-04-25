# Explanation for TT-014 — Embedding

The code in `solution.py` builds a simple embedding pipeline using PyTorch.

It performs two main tasks:

1. Create an embedding layer.
2. Convert token IDs into scaled embedding vectors.

## How the embedding layer is created

The `create_embedding_layer` function returns:

```python
nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
```

This layer stores a learnable vector of length `d_model` for each token ID from `0` to `vocab_size - 1`.

## How token embedding works

The `embed_tokens` function:

1. Accepts an `nn.Embedding` layer.
2. Takes a tensor of token IDs.
3. Looks up the embedding vector for each token using `embedding(tokens)`.
4. Multiplies the result by `math.sqrt(d_model)`.

If the input token tensor has shape `(N,)`, the output has shape `(N, d_model)`.
If the input token tensor has shape `(B, T)`, the output has shape `(B, T, d_model)`.

## Why scaling is used

Multiplying by `sqrt(d_model)` is a common transformer convention. It increases the embedding magnitude so that token embeddings remain on a compatible scale when combined with other components such as positional encodings.

## Notes on behavior

- The embedding weights are learnable parameters.
- Token IDs must be valid integer indices for the embedding table.
- The function does not perform extra validation on token range or tensor dtype.
- The implementation is short because PyTorch handles the lookup internally.
