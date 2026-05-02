# Explanation for TT-020 - Encoder Block (NumPy)

A Transformer encoder block processes a sequence by mixing information across tokens with self-attention, then refining each token representation with a feed-forward network.

## Block Formula

```text
attention_output = MultiHeadAttention(x, x, x)
z1 = LayerNorm(x + attention_output)
ffn_output = FeedForward(z1)
z2 = LayerNorm(z1 + ffn_output)
```

The final output `z2` has the same shape as the input `x`.

## Steps in `solution.py`

### 1. Multi-Head Self-Attention
```python
mha_output = multi_head_attention(
    Q=x, K=x, V=x,
    W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o,
    num_heads=num_heads
)
```

Because this is encoder self-attention, the same input `x` is used as query, key, and value.

Internally, the function:

- projects Q, K, and V
- splits them into heads
- computes scaled dot-product attention
- combines the heads
- applies the output projection

### 2. First Residual Connection and LayerNorm
```python
first_layer = layer_norm(mha_output + x, gamma1, beta1, eps=1e-6)
```

The residual connection adds the original input back to the attention result. LayerNorm then stabilizes the features across the last dimension.

### 3. Feed-Forward Network
```python
hidden = feed_forward(first_layer, W1=W1, b1=b1, W2=W2, b2=b2)
```

The feed-forward network is applied independently to each token:

```text
FFN(x) = ReLU(xW1 + b1)W2 + b2
```

### 4. Second Residual Connection and LayerNorm
```python
second_layer = layer_norm(first_layer + hidden, gamma2, beta2, eps=1e-6)
```

The feed-forward output is added back to its input, then normalized again.

## Shape Flow

For input:

```python
x.shape == (batch_size, seq_len, d_model)
```

The main tensors keep this shape:

```text
mha_output   -> (batch_size, seq_len, d_model)
first_layer  -> (batch_size, seq_len, d_model)
hidden       -> (batch_size, seq_len, d_model)
second_layer -> (batch_size, seq_len, d_model)
```

## Why Residual Connections Matter

Residual connections allow the block to preserve information from earlier representations. They also make deeper Transformer stacks easier to optimize because each sublayer only needs to learn a useful transformation on top of the existing representation.

## Why LayerNorm Matters

LayerNorm keeps each token representation numerically stable by normalizing across the feature dimension. This is especially useful in Transformers because sequence lengths can vary and batch statistics are less reliable than per-token feature statistics.

## Complexity

- Multi-head attention time: O(batch x seq_len^2 x d_model)
- Feed-forward time: O(batch x seq_len x d_model x d_ff)
- Output space: O(batch x seq_len x d_model)
