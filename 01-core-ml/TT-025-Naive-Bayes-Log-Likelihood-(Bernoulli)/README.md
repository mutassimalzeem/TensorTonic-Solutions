# TT-025 - Naive Bayes Log Likelihood (Bernoulli)

## Problem Link
TensorTonic: Naive Bayes Log Likelihood (Bernoulli)

> Note: The exact problem statement is on TensorTonic. This file documents the implementation used in `solution.py`.

---

## Problem Summary

Implement **Bernoulli Naive Bayes** log-likelihood scoring using NumPy. Given binary training features, training labels, and binary test features, compute the log score for each test sample under each class.

For each class, Bernoulli Naive Bayes estimates:

```python
theta = P(feature = 1 | class)
```

Then each test sample is scored with:

```python
log P(class) + sum(x * log(theta) + (1 - x) * log(1 - theta))
```

---

## Official Problem Statement
Implement a NumPy-based Bernoulli Naive Bayes helper that returns class log-likelihood scores for test samples.

---

## Constraints
- `X_train` should be a 2D binary feature matrix.
- `y_train` should contain the class label for each training sample.
- `X_test` should be a 2D binary feature matrix.
- `alpha` is the Laplace smoothing value.
- Classes are processed in sorted order from `np.unique(y_train)`.
- Return a NumPy array with shape `(n_test, n_classes)`.
- The returned values are log scores, not final class labels.

---

## Example

### Input
```python
X_train = [[1, 0], [1, 1], [0, 1], [0, 0]]
y_train = [1, 1, 0, 0]
X_test = [[1, 0], [0, 1]]

scores = naive_bayes_bernoulli(X_train, y_train, X_test, alpha=1.0)
```

### Output
```python
array([[-2.77258872, -1.67397643],
       [-1.67397643, -2.77258872]])
```

Class order is `[0, 1]`. The first test sample scores higher for class `1`, and the second test sample scores higher for class `0`.

---

## Solution Overview

The implementation in `solution.py`:

- converts training and test features to NumPy float arrays
- extracts sorted class labels with `np.unique`
- loops through each class
- builds a boolean mask for samples in that class
- computes smoothed Bernoulli probabilities for each feature
- stores `log(theta)` and `log(1 - theta)`
- computes log class priors
- converts the collected log values into NumPy arrays
- scores all test samples using matrix multiplication
- returns the log-likelihood score matrix

---

## Usage

```python
from solution import naive_bayes_bernoulli

X_train = [[1, 0], [1, 1], [0, 1], [0, 0]]
y_train = [1, 1, 0, 0]
X_test = [[1, 0], [0, 1]]

scores = naive_bayes_bernoulli(X_train, y_train, X_test)
print(scores)
```
