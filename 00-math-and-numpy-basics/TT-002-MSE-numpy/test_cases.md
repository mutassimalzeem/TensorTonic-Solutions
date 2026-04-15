# Test Cases for TT-002 — Mean Squared Error (NumPy)

## Basic Cases

1. Matching arrays
```python
y_pred = [1.0, 2.0, 3.0]
y_true = [1.0, 2.0, 3.0]
# output: 0.0
```

2. Simple error
```python
y_pred = [2.0, 3.0, 4.0]
y_true = [1.0, 2.0, 3.0]
# output: 1.0
```

## Different shapes

3. Mismatched lengths
```python
y_pred = [1.0, 2.0]
y_true = [1.0, 2.0, 3.0]
# output: None
```

4. Different shapes with arrays
```python
y_pred = np.array([1.0, 2.0])
y_true = np.array([[1.0, 2.0]])
# output: None
```

## Scalar inputs

5. Scalars
```python
y_pred = 2.0
y_true = 1.0
# output: 1.0
```

## Array inputs of different numeric types

6. Integer and float values
```python
y_pred = [1, 2, 3]
y_true = [0.0, 2.0, 1.0]
# output: 0.6666666666666666
```

## Notes
- The function should preserve compatibility between Python lists and NumPy arrays by converting inputs to arrays first.
- When shapes mismatch, the function returns `None` rather than raising an exception.
