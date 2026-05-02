import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    positions = np.arange(seq_length)[:, np.newaxis]

    dims = np.arange(0, d_model, 2)[np.newaxis, :]

    angles = positions / (10000 ** (dims / d_model))

    PE = np.zeros((seq_length, d_model))

    # Fill even columns with sin, odd columns with cos
    PE[:, 0::2] = np.sin(angles)  # 0, 2, 4, ...
    PE[:, 1::2] = np.cos(angles)  # 1, 3, 5, ...

    return PE

