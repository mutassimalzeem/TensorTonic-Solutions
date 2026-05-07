# Notes for TT-025 - Naive Bayes Log Likelihood (Bernoulli)

- Bernoulli Naive Bayes is for binary features.
- This helper returns log scores, not predicted labels.
- `np.unique(y_train)` controls the output class order.
- Laplace smoothing prevents `log(0)`.
- The smoothing denominator uses `n_c + 2 * alpha` because each feature can be `0` or `1`.
- `log_theta` is used when a test feature is `1`.
- `log_one_minus_theta` is used when a test feature is `0`.
- Log priors account for class frequency in the training set.
- The final scoring step is vectorized with matrix multiplication.
- The output shape is `(n_test, n_classes)`.
- The predicted class is the column with the largest log score.
