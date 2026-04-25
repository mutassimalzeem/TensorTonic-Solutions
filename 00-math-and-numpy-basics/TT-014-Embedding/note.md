# Notes for TT-014 — Embedding

- The implementation uses PyTorch instead of NumPy because `nn.Embedding` is a PyTorch module.
- `create_embedding_layer(vocab_size, d_model)` creates a learnable lookup table.
- Each token ID maps to a vector of length `d_model`.
- `embed_tokens` applies the embedding layer directly to the token tensor.
- The returned embedding vectors are scaled by `math.sqrt(d_model)`.
- This scaling is commonly used in transformer architectures.
- The shape of the output is the input token shape plus one extra dimension of size `d_model`.
- Token IDs must be within the valid vocabulary range supported by the embedding layer.
