# Notes for TT-024 - PCA Projection

- PCA projects data onto directions of maximum variance.
- The data must be centered before computing covariance.
- `np.mean(X, axis=0)` computes one mean per feature.
- The covariance matrix shape is `(n_features, n_features)`.
- Eigenvalues represent variance explained by each principal direction.
- Eigenvectors represent the principal directions.
- Sorting eigenvalues in descending order gives the most important components first.
- `W = sorted_eigenvectors[:, :k]` keeps the top `k` principal components.
- The projected output has shape `(n_samples, k)` before conversion to a list.
- Eigenvector signs are arbitrary, so projections can sometimes be sign-flipped and still be correct.
- Runtime is mainly controlled by covariance computation and eigen decomposition.
