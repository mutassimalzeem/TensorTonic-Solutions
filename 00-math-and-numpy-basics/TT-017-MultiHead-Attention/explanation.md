# Explanation for TT-017 — Multi-Head Attention (NumPy)

Multi-head attention is a key component of the Transformer architecture. It allows the model to jointly attend to information from different representation subspaces at different positions.

## Formula

For each head `h`:
```
head_h = Attention(QW_qh, KW_kh, VW_vh)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_o
```

Where `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`

## Steps in `solution.py`:

### 1. Linear Projections
```python
Q = Q @ W_q
K = K @ W_k
V = V @ W_v
```
Projects input into query, key, value spaces.

### 2. Split into Heads
```python
d_k = d_model // num_heads
Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
```
- Reshape: `(batch, seq, d_model)` → `(batch, seq, num_heads, d_k)`
- Transpose: `(batch, num_heads, seq, d_k)` — heads become parallel

### 3. Attention Scores
```python
scores = Q @ K.transpose(0, 1, 3, 2)
```
- Q: `(batch, heads, seq, d_k)`
- K^T: `(batch, heads, d_k, seq)`
- scores: `(batch, heads, seq, seq)`

### 4. Scale
```python
scores = scores / np.sqrt(d_k)
```
Prevents dot products from growing too large.

### 5. Softmax
```python
attention_weights = softmax(scores, axis=-1)
```
Stable softmax along the key dimension (last axis).

### 6. Weighted Sum
```python
attention = attention_weights @ V
```
- Result: `(batch, heads, seq, d_k)`

### 7. Combine Heads
```python
attention = attention.transpose(0, 2, 1, 3)
attention = attention.reshape(batch_size, seq_len, d_model)
```
- Transpose: `(batch, seq, heads, d_k)`
- Reshape: `(batch, seq, d_model)` — concatenate all heads

### 8. Output Projection
```python
output = attention @ W_o
```
Final linear projection mixing information across heads.

## Output Shape

`(batch_size, seq_len, d_model)` — same as input Q shape.

## Complexity

- Time: O(batch × seq² × d_model) — dominated by attention scores
- Space: O(batch × heads × seq²) — attention weight matrices

