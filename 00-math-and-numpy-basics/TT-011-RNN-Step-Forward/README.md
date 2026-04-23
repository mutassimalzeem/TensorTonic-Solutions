# TT-011 — RNN Step Forward (NumPy)

## Problem Link
TensorTonic: RNN Step Forward (NumPy)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **single RNN step forward**: h_t = tanh(x_t @ Wx + h_{t-1} @ Wh + b)

No batching, single time-step.

---

## Official Problem Statement
Compute RNN cell for one timestep. Inputs vectors x_t (D,), h_prev (H,). Weights Wx (D,H), Wh (H,H), b (H,).

---

## Constraints
- Inputs: scalars/lists/arrays; auto-convert
- Defaults zero-init if scalar 0
- Output: 1D array (H,)
- No in-place mods
- No batch dim

---

## Example

### Input
```python
x_t = [0.0, 0.0]
h_prev = [1.0, -1.0]
Wx = 0
Wh = np.eye(2)
b = 0
```

### Output ≈
```python
array([ 0.76159416, 0.        ])
```

---

## Solution Overview

In `solution.py`:
- `np.atleast_1d` + float convert
- Handle scalar-zero defaults
- `preact = x_t @ Wx + h_prev @ Wh + b`
- `h_t = np.tanh(preact)`

Robust input handling.

---

## Usage

```python
import numpy as np
from solution import rnn_step_forward

out = rnn_step_forward([0.0,0.0], [1.0,-1.0], 0, np.eye(2), 0)
print(out)
```

