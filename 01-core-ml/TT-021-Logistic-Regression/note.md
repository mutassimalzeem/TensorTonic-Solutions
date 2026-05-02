# Notes for TT-021 - Logistic Regression

- Logistic regression is a linear classifier for binary labels.
- The sigmoid function maps raw logits into probabilities between `0` and `1`.
- This implementation trains with batch gradient descent.
- `lr` controls the update size; too large can make training unstable.
- `steps` controls how many gradient updates are performed.
- The implementation clips probabilities before computing binary cross entropy to avoid `log(0)`.
- `X` must be 2D, even for a single feature, such as `[[0], [1], [2]]`.
- `y` should contain binary labels: `0` and `1`.
- The returned `w` contains one weight per feature, and `b` is the intercept term.
- A prediction can be made with `(_sigmoid(X @ w + b) >= 0.5)`.
