import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mean_x = np.mean(x, axis= -1, keepdims = True)

    var = np.var(x, axis = -1, keepdims = True)
    norm = (x - mean_x) / np.sqrt(var + eps)
    layer_norm_output = gamma * norm + beta

    return layer_norm_output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch_size = Q.shape[0]
    seq_len = Q.shape[1]

    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    d_model = Q.shape[-1]
    d_k = d_model // num_heads

    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    score = Q @ K.transpose(0, 1, 3, 2)
    scores = score / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    attention = attention_weights @ V

    attention = attention.transpose(0, 2, 1, 3)
    attention = attention.reshape(batch_size, seq_len, d_model)

    # 9. Output projection
    multiHead_output = attention @ W_o

    return multiHead_output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden1 = np.dot(x, W1) + b1
    relu = np.maximum(0, hidden1)
    feed_forward_output = np.dot(relu, W2) + b2
    return feed_forward_output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    
    # 1. Multi-Head Self-Attention (Q, K, and V are all 'x')
    mha_output = multi_head_attention(
        Q=x, K=x, V=x, 
        W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, 
        num_heads=num_heads
    )

    # 2. Add and Norm 1
    first_layer = layer_norm((mha_output + x), gamma1, beta1, eps=1e-6)

    # 3. Feed Forward
    hidden = feed_forward(first_layer, W1=W1, b1=b1, W2=W2, b2=b2)
    
    # 4. Add and Norm 2
    second_layer = layer_norm((first_layer + hidden), gamma2, beta2, eps=1e-6)

    return second_layer