import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Simulate any random rectangular matrix A
A = np.random.rand(6, 3)
print(A)

# 1.1 Rank and trace of A
print("The Rank of a Matrix: ", np.linalg.matrix_rank(A))


# 1.1 Trace of A
print("Trace of A: ", np.trace(A))

# 1.2 What is the determinant of A?
if A.shape[0] == A.shape[1]:  # Compare rows with columns
    print("The determinant of a Matrix: ", np.linalg.det(A))
else:
    print("The matrix is rectangular, determinant couldn't be calculated")

# 1.3 Can you invert A? How?
# We can use the pseudoinverse of a rectangular matrix, which is a generalization of the inverse for non-square matrices.
print(np.linalg.pinv(A))

#1.4 How are eigenvalues and eigenvectors of A’A and AA’ related? What interesting differences can you notice between both?
print("The eigvals of A'A: ", np.linalg.eigvals(A.T @ A))
print("The eigvals of AA': ", np.linalg.eigvals(A @ A.T))
