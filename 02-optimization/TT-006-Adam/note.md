# Notes for TT-006 - Adam

- Adam stands for Adaptive Moment Estimation.
- It combines momentum and RMSProp-style adaptive scaling.
- `m` stores the exponentially weighted average of gradients.
- `v` stores the exponentially weighted average of squared gradients.
- `m_hat` and `v_hat` are bias-corrected versions of `m` and `v`.
- `eps` prevents division by zero and improves numerical stability.
- `t` should start at `1`, not `0`, because bias correction uses `1 - beta ** t`.
- The function returns updated `param`, `m`, and `v`.
- The updated `m` and `v` are optimizer state values and should be reused in the next step.
- The current `solution.py` includes a demo `print(...)` call after the function definition.
