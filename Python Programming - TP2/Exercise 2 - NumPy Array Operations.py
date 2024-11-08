# EXERCISE 2
import numpy as np

# a
matrix1 = np.random.randint(1, 11, size=(3, 3))
matrix2 = np.random.randint(1, 11, size=(3, 3))

print(f"Matrix 1:\n {matrix1}")
print(f"Matrix 2:\n {matrix2} \n")

# b
addition = matrix1 + matrix2
multiplication = matrix1 @ matrix2
transpose = matrix1.T
inverse = np.linalg.inv(matrix1)

print(f"\nAddition:\n {addition}")
print(f"\nMultiplication:\n {multiplication}")
print(f"\nTranspose:\n {transpose}")
print(f"\nInverse:\n {inverse} \n")

# c
total_sum = np.sum(matrix1)
row_means = np.mean(matrix1, axis=1)
col_maxes = np.max(matrix1, axis=0)

print(f"Sum of all elements: {total_sum}")
print(f"Mean of each row: {row_means}")
print(f"Maximum value in each column: {col_maxes}")



'''
[Running] python -u "h:\DAISI\Python Programming - TP2\Exercise 2 - NumPy Array Operations.py"
Matrix 1:
 [[2 3 1]
 [5 5 3]
 [6 1 8]]
Matrix 2:
 [[7 7 8]
 [9 4 8]
 [2 4 6]] 


Addition:
 [[ 9 10  9]
 [14  9 11]
 [ 8  5 14]]

Multiplication:
 [[ 43  30  46]
 [ 86  67  98]
 [ 67  78 104]]

Transpose:
 [[2 5 6]
 [3 5 1]
 [1 3 8]]

Inverse:
 [[-2.17647059  1.35294118 -0.23529412]
 [ 1.29411765 -0.58823529  0.05882353]
 [ 1.47058824 -0.94117647  0.29411765]] 

Sum of all elements: 34
Mean of each row: [2.         4.33333333 5.        ]
Maximum value in each column: [6 5 8]

[Done] exited with code=0 in 2.451 seconds
'''