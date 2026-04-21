import numpy as np

def matrix_transpose(A):
    rows = np.shape(A)[0]
    cols = np.shape(A)[1]

    B = [[0 for j in range(rows)] for i in range (cols)]

    for i in range(rows):
        for j in range(cols):
            B[j][i] = A[i][j]

    return np.array(B)