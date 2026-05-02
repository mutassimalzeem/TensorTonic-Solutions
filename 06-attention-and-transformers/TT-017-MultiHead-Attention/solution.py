import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    # Q shape: (batch_size, seq_len, d_model)
    batch_size = Q.shape[0]
    seq_len = Q.shape[1]

    # 1. Linear projections
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    # 2. Dimensions after projection
    d_model = Q.shape[-1]
    d_k = d_model // num_heads

    # 3. Split into heads
    # From: (batch_size, seq_len, d_model)
    # To:   (batch_size, num_heads, seq_len, d_k)
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # 4. Attention scores
    # Q: (batch_size, num_heads, seq_len, d_k)
    # K.T: (batch_size, num_heads, d_k, seq_len)
    scores = Q @ K.transpose(0, 1, 3, 2)

    # 5. Scale
    scores = scores / np.sqrt(d_k)

    # 6. Softmax
    attention_weights = softmax(scores, axis=-1)

    # 7. Weighted sum
    attention = attention_weights @ V

    # 8. Combine heads
    # From: (batch_size, num_heads, seq_len, d_k)
    # To:   (batch_size, seq_len, d_model)
    attention = attention.transpose(0, 2, 1, 3)
    attention = attention.reshape(batch_size, seq_len, d_model)

    # 9. Output projection
    output = attention @ W_o

    return output