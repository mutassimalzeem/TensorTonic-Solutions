# Notes for TT-005 — Gradient Descent

- The function optimizes the quadratic expression `ax^2 + bx + c`.
- The derivative used in the update is `2ax + b`.
- The constant term `c` does not affect the gradient.
- The current implementation overrides the input learning rate with `1 / (2 * a)`.
- For any `steps > 0`, the returned result is `-b / (2a)`, the minimizer of the quadratic.
- The function returns the final `x` position, not the minimum function value.
- If `a == 0`, the expression `1 / (2 * a)` is undefined, so the current implementation requires `a != 0`.
