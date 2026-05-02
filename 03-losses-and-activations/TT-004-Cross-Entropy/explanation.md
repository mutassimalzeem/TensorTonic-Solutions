# Explanation for TT-004 — Cross Entropy

Cross entropy measures how well the predicted probability distribution matches the true distribution.

For one-hot encoded labels, only the predicted probability for the true class contributes to the loss. If `y_true` is a 1D array of class indices, the implementation extracts the correct class probabilities from `y_pred`.

## Formula

For a single sample with one-hot label `y_true` and prediction `y_pred`:

```
loss_i = -sum(y_true * log(y_pred))
```

For a batch of samples, the final loss is the mean of each sample's cross entropy:

```
loss = -mean(sum(y_true * log(y_pred), axis=1))
```

## Implementation details

- `np.array(y_true)` converts labels to a NumPy array.
- `np.array(y_pred, dtype=np.float64)` converts predicted probabilities to a NumPy array.
- `np.clip(y_pred, 1e-15, 1 - 1e-15)` avoids `log(0)` and numeric instability.
- If `y_true` and `y_pred` have matching shapes, the implementation computes one-hot cross entropy.
- If `y_true` is a 1D integer label array, the implementation indexes the correct class probabilities from `y_pred`.
- `np.mean(...)` averages across all samples.

## Notes

- `y_true` should be one-hot encoded, e.g. `[1, 0, 0]`.
- `y_pred` should be valid probabilities, not raw logits.
- If `y_pred` contains zeros or values outside `(0, 1)`, clipping prevents invalid log values.
- The output is a single scalar representing the mean loss for the batch.
