# Explanation for TT-007 — Matrix Transposition (NumPy)

Matrix transposition converts A (m rows × n cols) to B (n rows × m cols) where B[j][i] = A[i][j].

## Steps in `solution.py`:

1. Extract dimensions: `rows = np.shape(A)[0]`, `cols = np.shape(A)[1]`
2. Initialize `B = [[0 for j in range(rows)] for i in range(cols)]` – list of cols lists, each with rows zeros.
3. Nested loops:
   ```
   for i in range(rows):
       for j in range(cols):
           B[j][i] = A[i][j]
   ```
4. Convert `return np.array(B)` – shape becomes (cols, rows)

## Notes on behavior
- Time complexity: O(m*n) due to loops.
- Manual implementation avoids `A.T` for educational purposes.
- Output preserves data types from input.
- Assumes input is rectangular 2D array (no checks for non-2D/empty).

## Output shape
Input shape (m, n) → Output (n, m). E.g., (2,3) → (3,2).
