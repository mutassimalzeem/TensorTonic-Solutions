# Test Cases for TT-024 - PCA Projection

## Case 1: One-dimensional variation

```python
X = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
result = pca_projection(X, k=1)
```

Expected:
```python
[[-2.0], [-1.0], [0.0], [1.0], [2.0]]
```

The sign may be flipped depending on the eigenvector direction.

## Case 2: Keep two components

```python
X = [[1, 2], [3, 4], [5, 6]]
result = pca_projection(X, k=2)
```

Expected:
```python
len(result) == 3
len(result[0]) == 2
```

The output keeps the same number of rows and returns two projected features.

## Case 3: Centered projection has zero mean

```python
X = [[1, 1], [2, 2], [3, 3]]
result = pca_projection(X, k=1)
```

Expected:
```python
np.mean(result, axis=0) is close to [0.0]
```

PCA projection is performed after centering the input data.

## Case 4: Reduce three features to one

```python
X = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]
result = pca_projection(X, k=1)
```

Expected:
```python
len(result) == 4
len(result[0]) == 1
```

Only the strongest principal component is kept.

## Case 5: Return type

```python
X = [[0, 1], [1, 0], [2, 1]]
result = pca_projection(X, k=1)
```

Expected:
```python
isinstance(result, list)
isinstance(result[0], list)
```

The implementation converts the NumPy projection to a nested Python list before returning.
