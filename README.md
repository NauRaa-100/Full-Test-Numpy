## NumPy Review 

This repository contains a comprehensive review of NumPy basics and advanced operations, with a focus on applications in Artificial Intelligence (AI) and Machine Learning (ML).

Topics Covered
1. Array Creation

np.array() → Convert list/tuple into an array.

np.zeros() → Create an array filled with zeros.

np.ones() → Create an array filled with ones.

np.full() → Create an array filled with a constant value.

np.eye() → Identity matrix.

np.arange(start, stop, step) → Range of numbers.

np.linspace(start, stop, num) → Linearly spaced values.

np.random.rand() → Random values between 0 and 1.

np.random.randn() → Random values from a normal distribution.

np.random.randint(low, high, size) → Random integers.

2. Array Attributes

arr.shape → Shape (rows, columns).

arr.ndim → Number of dimensions.

arr.size → Number of elements.

arr.dtype → Data type (int, float, etc.).

3. Indexing & Slicing

arr[0], arr[-1] → Single element.

arr[1:4] → Slice.

arr[:, 0] → First column.

arr[0, :] → First row.

arr[::2] → Step slicing.

Boolean Indexing → arr[arr > 5]
Fancy Indexing → arr[[0,2,4]]

4. Math Operations

Arithmetic: + - * /

Power: arr ** 2

Universal functions:

np.sqrt, np.exp, np.log

np.sin, np.cos, np.tan

5. Statistics

np.mean() → Mean.

np.median() → Median.

np.std() → Standard deviation.

np.var() → Variance.

np.min(), np.max() → Min & Max.

np.argmax(), np.argmin() → Indices of max/min.

6. Matrix Operations

arr.T → Transpose.

np.dot(A, B) or A @ B → Matrix multiplication.

np.matmul(A, B) → Matrix multiplication.

np.linalg.inv(A) → Inverse.

np.linalg.det(A) → Determinant.

np.linalg.eig(A) → Eigenvalues & Eigenvectors.

7. Stacking & Splitting

np.concatenate([A, B], axis=0/1)

np.vstack([A, B]) → Vertical stack.

np.hstack([A, B]) → Horizontal stack.

np.split(arr, n) → Split array.

8. Reshaping

arr.reshape((rows, cols))

arr.flatten() → Flatten to 1D.

arr.ravel() → Efficient flatten.

9. Broadcasting

Automatic expansion of smaller arrays.

Example:

A = np.array([[1,2,3],[4,5,6]])
B = np.array([1,2,3])
A + B  # Adds B to each row

10. AI/ML Applications

Data normalization / standardization:

(arr - arr.mean()) / arr.std()


Random weights for neural networks:
np.random.randn()

Matrix multiplication: Forward & Backpropagation.

PCA: Eigenvalues & Eigenvectors.

Reshaping images: (28x28) → (784,).

📚 References

NumPy Documentation

Scipy Lecture Notes

This README summarizes all the essential operations we covered during the NumPy review session.

