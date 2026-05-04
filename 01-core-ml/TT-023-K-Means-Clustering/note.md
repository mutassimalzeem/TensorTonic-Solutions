# Notes for TT-023 - K-Means Clustering

- This helper performs the assignment step of K-means clustering.
- It returns centroid indices, not updated centroid coordinates.
- Squared Euclidean distance is used for comparison.
- Skipping `sqrt` is valid because it preserves nearest-distance ordering.
- The output is a flat Python list of integers.
- The output length is equal to the number of input points.
- If distances tie, `np.argmin` chooses the first nearest centroid.
- Runtime is `O(n_points * n_centroids * d)`.
- This implementation is clear and direct; a vectorized distance matrix could be faster for large datasets.
