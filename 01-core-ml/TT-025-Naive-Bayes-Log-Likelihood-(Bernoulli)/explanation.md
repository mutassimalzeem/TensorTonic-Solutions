# Explanation for TT-025 - Naive Bayes Log Likelihood (Bernoulli)

Bernoulli Naive Bayes is used when features are binary, usually `0` or `1`. It models whether each feature is present or absent for each class.

This implementation returns log-likelihood scores for every test sample and every class. It does not directly return predicted class labels.

## Class setup

The code converts inputs first:

```python
X_train = np.asarray(X_train, dtype=np.float64)
y_train = np.asarray(y_train)
X_test  = np.asarray(X_test,  dtype=np.float64)
```

Then it finds the class labels:

```python
classes = np.unique(y_train)
```

`np.unique` sorts labels, so the output columns follow that sorted class order.

## Laplace smoothing

For each class `c`, the implementation builds a mask:

```python
mask = (y_train == c)
```

Then it computes the Bernoulli probability for each feature:

```python
theta = (X_train[mask].sum(axis=0) + alpha) / (n_c + 2.0 * alpha)
```

The `+ alpha` prevents probabilities from becoming exactly `0`. The denominator uses `2.0 * alpha` because a Bernoulli feature has two possible outcomes: `0` and `1`.

## Log probabilities

The code stores two log-probability arrays:

```python
log_theta.append(np.log(theta))
log_one_minus_theta.append(np.log(1.0 - theta))
```

For a feature value of `1`, the model uses `log(theta)`. For a feature value of `0`, it uses `log(1 - theta)`.

The class prior is:

```python
log_priors.append(np.log(n_c / n_train))
```

## Vectorized scoring

After collecting all class parameters, the implementation scores test samples with:

```python
X_test @ log_theta.T + (1.0 - X_test) @ log_one_minus_theta.T + log_priors
```

This matches the Bernoulli Naive Bayes log score:

```python
log P(class) + sum(x * log(theta) + (1 - x) * log(1 - theta))
```

## Output shape

For `m = len(X_test)` and `c = number of classes`, the output shape is:

```python
(m, c)
```

Each row contains the log scores for one test sample. Each column corresponds to one class in sorted class order.

## Prediction note

To convert log scores into class predictions, take the class with the highest score:

```python
classes[np.argmax(scores, axis=1)]
```
