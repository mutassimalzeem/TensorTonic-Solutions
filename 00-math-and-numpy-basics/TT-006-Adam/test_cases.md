# Test Cases for TT-006 - Adam

## Case 1: Zero gradient

```python
param = [1.0, 2.0]
grad = [0, 0]
m = [0, 0]
v = [0, 0]
t = 1
lr = 0.001
```

Expected:

```python
(array([1., 2.]), array([0., 0.]), array([0., 0.]))
```

With zero gradient, the parameter does not change.

## Case 2: Basic vector update

```python
param = [1.0, 2.0]
grad = [0.1, -0.2]
m = [0, 0]
v = [0, 0]
t = 1
lr = 0.001
```

Expected:

```python
(array([0.999, 2.001]), array([ 0.01, -0.02]), array([1.e-05, 4.e-05]))
```

Positive gradients reduce parameters, while negative gradients increase them.

## Case 3: Existing optimizer state

```python
param = [1.0]
grad = [0.5]
m = [0.1]
v = [0.2]
t = 2
lr = 0.01
```

Expected:

```python
(array([0.99926343]), array([0.14]), array([0.20005]))
```

This checks that previous `m` and `v` values are included in the next update.

## Case 4: Larger learning rate

```python
param = [3.0, -1.0]
grad = [1.0, 1.0]
m = [0.0, 0.0]
v = [0.0, 0.0]
t = 1
lr = 0.1
```

Expected:

```python
(array([ 2.9, -1.1]), array([0.1, 0.1]), array([0.001, 0.001]))
```

Both parameters move in the negative gradient direction.

## Case 5: Small gradients

```python
param = [1.0, 2.0, 3.0]
grad = [0.01, 0.02, 0.03]
m = [0, 0, 0]
v = [0, 0, 0]
t = 1
lr = 0.001
```

Expected:

```python
(array([0.99900005, 1.99900001, 2.99900001]), array([0.001, 0.002, 0.003]), array([1.e-07, 4.e-07, 9.e-07]))
```

The parameter update remains small because the learning rate is small.
