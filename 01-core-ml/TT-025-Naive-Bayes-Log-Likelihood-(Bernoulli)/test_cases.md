# Test Cases for TT-025 - Naive Bayes Log Likelihood (Bernoulli)

## Case 1: Balanced two-class data

```python
X_train = [[1, 0], [1, 1], [0, 1], [0, 0]]
y_train = [1, 1, 0, 0]
X_test = [[1, 0], [0, 1]]
result = naive_bayes_bernoulli(X_train, y_train, X_test, alpha=1.0)
```

Expected:
```python
array([[-2.77258872, -1.67397643],
       [-1.67397643, -2.77258872]])
```

Class order is `[0, 1]`.

## Case 2: Output shape

```python
X_train = [[1, 0], [0, 1], [1, 1]]
y_train = [0, 1, 1]
X_test = [[1, 1], [0, 0], [1, 0]]
result = naive_bayes_bernoulli(X_train, y_train, X_test)
```

Expected:
```python
result.shape == (3, 2)
```

There are three test samples and two classes.

## Case 3: Smoothing avoids log zero

```python
X_train = [[1, 1], [1, 1], [0, 0]]
y_train = [1, 1, 0]
X_test = [[0, 1]]
result = naive_bayes_bernoulli(X_train, y_train, X_test, alpha=1.0)
```

Expected:
```python
np.isfinite(result).all()
```

Even if a feature is never seen with a class, Laplace smoothing keeps the log score finite.

## Case 4: Sorted class order

```python
X_train = [[1, 0], [0, 1], [1, 1]]
y_train = [5, 2, 5]
X_test = [[1, 1]]
result = naive_bayes_bernoulli(X_train, y_train, X_test)
```

Expected:
```python
result.shape == (1, 2)
columns correspond to classes [2, 5]
```

`np.unique(y_train)` sorts the class labels.

## Case 5: Convert scores to predictions

```python
X_train = [[1, 0], [1, 1], [0, 1], [0, 0]]
y_train = np.array([1, 1, 0, 0])
X_test = [[1, 0], [0, 1]]
scores = naive_bayes_bernoulli(X_train, y_train, X_test)
classes = np.unique(y_train)
predictions = classes[np.argmax(scores, axis=1)]
```

Expected:
```python
array([1, 0])
```
