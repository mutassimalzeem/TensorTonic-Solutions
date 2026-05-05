import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.array(X, dtype = float)

    #   Center Data
    # x_mean1 = 0
    # for i in range(np.shape(X)[0]):
    #     x_mean1 += (X[i][0])

    # x_mean2 = 0
    # for i in range(np.shape(X)[0]):
    #     x_mean2 += (X[i][1])

    # x_mean = [x_mean1, x_mean2]
    x_mean = np.mean(X, axis = 0)

    #   Compute Covariance Matrix
    x_c = X - x_mean
    c = (1 / (np.shape(X)[0] - 1)) * np.transpose(x_c) @ x_c

    #   Eigen Decomposition
    eigenvalues, eigenvectors = np.linalg.eig(c)
    idx = np.argsort(eigenvalues)[::-1]          #   Sort eigenvalues descending
    sorted_eigenvectors = eigenvectors[:, idx]
    #top_k_values = np.sort(eigenvectors.flatten())[-k:]
    
    #   Slice the top k columns
    W =  sorted_eigenvectors[:, :k]
    x_proj = x_c @ W

    return x_proj.tolist()

print(pca_projection(X = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], k = 1))