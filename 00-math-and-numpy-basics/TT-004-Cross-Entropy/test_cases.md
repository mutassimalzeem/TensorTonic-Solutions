# Test Cases for TT-004 — Cross Entropy

## Case 1: Basic one-hot predictions

```python
y_true = [[1, 0, 0], [0, 1, 0]]
y_pred = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]
```

Expected: `0.1602501` (approximately)

## Case 2: Perfect predictions

```python
y_true = [[0, 1], [1, 0]]
y_pred = [[0.0, 1.0], [1.0, 0.0]]
```

Expected: `1e-15` or a very small positive value after clipping.

## Case 3: Uniform predictions

```python
y_true = [[1, 0, 0], [0, 0, 1]]
y_pred = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]
```

Expected: `1.0986123` (approximately)

## Case 4: Two samples, same class

```python
y_true = [[0, 1], [0, 1]]
y_pred = [[0.2, 0.8], [0.25, 0.75]]
```

Expected: `0.2876821` (approximately)

## Case 5: Sparse integer labels

```python
y_true = [0, 1]
y_pred = [[0.6, 0.4], [0.2, 0.8]]
```

Expected: `0.3566749` (approximately)
