# Test Cases for TT-023 - K-Means Clustering

## Case 1: Simple two-cluster assignment

```python
points = [[0, 0], [5, 5], [1, 1]]
centroids = [[0, 0], [5, 5]]
result = k_means_assignment(points, centroids)
```

Expected:
```python
[0, 1, 0]
```

## Case 2: Multiple points near different centroids

```python
points = [[-1, 0], [10, 9], [2, 2], [9, 11]]
centroids = [[0, 0], [10, 10]]
result = k_means_assignment(points, centroids)
```

Expected:
```python
[0, 1, 0, 1]
```

## Case 3: Three centroids

```python
points = [[0, 0], [5, 0], [0, 5], [4, 1]]
centroids = [[0, 0], [5, 0], [0, 5]]
result = k_means_assignment(points, centroids)
```

Expected:
```python
[0, 1, 2, 1]
```

## Case 4: Distance tie keeps first centroid

```python
points = [[1, 0]]
centroids = [[0, 0], [2, 0]]
result = k_means_assignment(points, centroids)
```

Expected:
```python
[0]
```

## Case 5: Empty point list

```python
points = []
centroids = [[0, 0], [1, 1]]
result = k_means_assignment(points, centroids)
```

Expected:
```python
[]
```
