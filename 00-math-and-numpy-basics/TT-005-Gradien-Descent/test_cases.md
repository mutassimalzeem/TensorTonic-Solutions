# Test Cases for TT-005 — Gradient Descent

## Case 1: Basic quadratic minimum

```python
a = 2
b = -8
c = 3
x0 = 0
lr = 0.1
steps = 5
```

Expected: `2.0`

Because `-b / (2a) = 8 / 4 = 2.0`.

## Case 2: Positive `b`

```python
a = 1
b = 6
c = 2
x0 = 10
lr = 0.5
steps = 3
```

Expected: `-3.0`

Because `-b / (2a) = -6 / 2 = -3.0`.

## Case 3: Zero steps

```python
a = 2
b = -8
c = 3
x0 = 7
lr = 0.1
steps = 0
```

Expected: `7`

With zero iterations, the function returns the original `x0`.

## Case 4: Different starting point, same optimum

```python
a = 4
b = -16
c = 1
x0 = -20
lr = 0.01
steps = 1
```

Expected: `2.0`

Because `-b / (2a) = 16 / 8 = 2.0`.

## Case 5: Constant term does not matter

```python
a = 3
b = -12
c = 999
x0 = 100
lr = 0.2
steps = 2
```

Expected: `2.0`

Because the derivative does not depend on `c`, and `-b / (2a) = 12 / 6 = 2.0`.
