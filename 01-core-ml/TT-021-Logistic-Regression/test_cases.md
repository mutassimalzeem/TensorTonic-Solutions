# Test Cases for TT-021 - Logistic Regression

## Case 1: Simple separable one-feature data

```python
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
w, b = train_logistic_regression(X, y, lr=0.1, steps=500)
```

Expected:
```python
w.shape == (1,)
b is a float
_sigmoid(np.dot(X, w) + b) gives lower probabilities for 0 and 1 than for 2 and 3
```

## Case 2: Already centered binary split

```python
X = [[-2], [-1], [1], [2]]
y = [0, 0, 1, 1]
w, b = train_logistic_regression(X, y, lr=0.1, steps=1000)
```

Expected:
```python
w[0] > 0
abs(b) is small
predictions == [0, 0, 1, 1] using threshold 0.5
```

## Case 3: Two features

```python
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]
w, b = train_logistic_regression(X, y, lr=0.1, steps=2000)
```

Expected:
```python
w.shape == (2,)
predictions[-1] is the largest probability
```

## Case 4: All-zero feature values

```python
X = [[0], [0], [0], [0]]
y = [0, 0, 1, 1]
w, b = train_logistic_regression(X, y, lr=0.1, steps=100)
```

Expected:
```python
w == array([0.])
b is close to 0.0
predicted probabilities are close to 0.5
```

## Case 5: No training steps

```python
X = [[0], [1], [2]]
y = [0, 1, 1]
w, b = train_logistic_regression(X, y, lr=0.1, steps=0)
```

Expected:
```python
w == array([0.])
b == 0.0
```
