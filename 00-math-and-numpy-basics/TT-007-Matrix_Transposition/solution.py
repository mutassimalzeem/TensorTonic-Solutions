import numpy as np

def matrix_transpose(A, B, row, col):
    
    for i in range(row):
        for j in range(col):
            return B[i][j] == A[j][i]