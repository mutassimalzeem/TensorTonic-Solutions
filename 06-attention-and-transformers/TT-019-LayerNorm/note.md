# Notes for TT-019 — Layer Normalization (NumPy)

- Normalizes across last dimension (features) per sample.
- Each sample gets independent mean/variance.
- gamma, beta are learnable (initialized to 1s and 0s typically).
- `eps=1e-6` prevents sqrt(0) for numerical stability.
- No bias term (already captured by mean subtraction).
- Unlike BatchNorm: independent of batch size.
- Preserves shape (input → output same).
- In Transformers: applied twice per layer (pre and post attention/FFN).
- Can be fused with linear layers in some implementations.
- PyTorch equivalent: `torch.nn.LayerNorm(normalized_shape)`.
- Recurrent networks: often applied before or after RNN.
- Supports any input dimensionality (2D, 3D, etc.).
- Used in BERT, GPT, ViT, etc.
- Gradient flows through gamma and beta for learned normalization.
