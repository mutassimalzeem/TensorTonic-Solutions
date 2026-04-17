# Notes for TT-004 — Cross Entropy

- This implementation expects `y_pred` to be probabilities for each class.
- Use a softmax layer before cross entropy if your model outputs logits.
- `np.clip` is used to keep `y_pred` values in `(1e-15, 1 - 1e-15)` and avoid `log(0)`.
- The loss is averaged across all samples, so the result is a scalar.
- `y_true` may be one-hot encoded or a 1D array of integer class labels.
