# Test Cases for TT-022 - KNN Distance

## Case 1: Single test point

```python
X_train = [[0, 0], [2, 0], [0, 2]]
X_test = [[1, 0]]
result = knn_distance(X_train, X_test, k=2)
```

Expected:
```python
array([[0, 1]])
```

## Case 2: Multiple test points

```python
X_train = [[0, 0], [5, 5], [1, 1]]
X_test = [[0, 1], [4, 5]]
result = knn_distance(X_train, X_test, k=1)
```

Expected:
```python
array([[0],
       [1]])
```

## Case 3: k larger than training set

```python
X_train = [[0, 0]]
X_test = [[1, 1]]
result = knn_distance(X_train, X_test, k=3)
```

Expected:
```python
array([[0, -1, -1]])
```

## Case 4: Empty test set

```python
X_train = [[0, 0]]
X_test = []
result = knn_distance(X_train, X_test, k=2)
```

Expected:
```python
result.shape == (0, 2)
result.dtype is an integer dtype
```

## Case 5: Distance tie keeps original order

```python
X_train = [[0, 0], [2, 0]]
X_test = [[1, 0]]
result = knn_distance(X_train, X_test, k=2)
```

Expected:
```python
array([[0, 1]])
```
