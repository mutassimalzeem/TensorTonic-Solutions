# Notes for TT-022 - KNN Distance

- This helper returns nearest neighbor indices, not predicted labels.
- Euclidean distance is used for comparing points.
- The output always has shape `(len(X_test), k)`.
- The output dtype is integer.
- Empty `X_test` returns `np.empty((0, k), dtype=int)`.
- If `k` is greater than the number of training points, missing entries are filled with `-1`.
- Distance ties keep the earlier training index first because Python sorting is stable.
- Runtime is `O(n_test * n_train * d)` for distance computation plus sorting per test point.
- This implementation is clear and direct; a fully vectorized version could be faster for large datasets.
