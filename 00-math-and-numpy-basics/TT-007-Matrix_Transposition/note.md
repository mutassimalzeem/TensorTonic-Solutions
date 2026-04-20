# Notes for TT-007 — Matrix Transposition (NumPy)

- Transpose swaps rows ↔ columns: element at position (i,j) moves to (j,i).
- Implementation uses list-of-lists for B (cols outer loops for rows).
- `np.shape(A)[0/1]` extracts dimensions safely.
- Double loop visits every element exactly once.
- Final `np.array(B)` converts to NumPy array (required output format).
- No input validation (assumes valid 2D rectangular np.array).
- Edge cases to test: 1x1 matrix, 1xN row vector, Nx1 column vector, square matrix, empty? (may fail).
- Purpose: Teach manual transpose vs NumPy's efficient `A.T` or `np.transpose(A)`.
- Time: O(m*n), Space: O(m*n) for new matrix.
