# Explanation for TT-005 — Gradient Descent

Gradient descent is an optimization method used to minimize a function by repeatedly moving in the direction opposite to its derivative.

For the quadratic function:

```python
f(x) = ax^2 + bx + c
```

the derivative is:

```python
f'(x) = 2ax + b
```

So a standard gradient descent update is:

```python
x = x - lr * (2ax + b)
```

## What the current implementation does

The implementation in `solution.py` performs these steps:

1. It defines a function named `gradient_descent_quadratic`.
2. It resets the learning rate to `1 / (2 * a)`.
3. It runs a loop for `steps` iterations.
4. Inside the loop, it applies the quadratic gradient expression `2 * a * x0 + b`.
5. It returns the final value of `x0`.

## Behavior of this implementation

- The parameter `c` does not affect the derivative, so it does not affect the update rule.
- The provided `lr` argument is overwritten by `1 / (2 * a)`.
- For any positive number of steps, the returned value becomes:

```python
-b / (2a)
```

which is the exact minimizer of the quadratic function.

## Why `-b / (2a)` is the minimum

The minimum of a quadratic function occurs where its derivative is zero:

```python
2ax + b = 0
```

Solving for `x`:

```python
x = -b / (2a)
```

That is why the implementation returns the optimum location for positive `steps`.

## Notes

- This exercise demonstrates the gradient formula for a quadratic function.
- The solution uses scalar arithmetic only.
- The function returns the optimized `x` value, not the function value `f(x)`.
- If `steps == 0`, the function returns the original `x0` input unchanged.
