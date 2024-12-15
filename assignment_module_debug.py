import numpy as np
from scipy.optimize import linear_sum_assignment
import copy


def preprocess_array(Arr):
    if all(i < 0 for i in Arr):
        print("All numbers are negative, adding the minimum number to all elements")
        min_num = min(Arr)
        Arr = [i - min_num for i in Arr]
    if any(i < 0 for i in Arr):
        print("There are negative and positive numbers in the array")
    print(Arr)
    return Arr

def create_cost_matrix(n, arr):
    
    cost_matrix = [[arr[i * n + j] for j in range(n)] for i in range(n)]
    print("[Input] Cost Matrix:")
    for row in cost_matrix:
        print(f"\t{row}")
    return cost_matrix

def reduce_matrix(cost_matrix, n):
    new_matrix = []
    for i in range(n):
        min_value = min(cost_matrix[i])
        new_matrix.append([value - min_value for value in cost_matrix[i]])
    non_zero_rows = []
    non_zero_cols = []
    for i in range(n):
        min_of_row = min(new_matrix[i])
        if min_of_row != 0:
            non_zero_rows.append(i)
    for j in range(n):
        col = [new_matrix[i][j] for i in range(n)]
        min_of_col = min(col)
        if min_of_col != 0:
            non_zero_cols.append(j)
    if non_zero_rows or non_zero_cols:
        for i in non_zero_rows:
            new_matrix[i] = [value - min(new_matrix[i]) for value in new_matrix[i]]
        for j in non_zero_cols:
            col = [new_matrix[i][j] for i in range(n)]
            min_of_col = min(col)
            for i in range(n):
                new_matrix[i][j] -= min_of_col
    print("[Output] New Matrix:")
    for row in new_matrix:
        print(f"\t{row}")
    return new_matrix

def count_zeros(matrix, n):
    row_zeros = [row.count(0) for row in matrix]
    col_zeros = [sum(1 for row in matrix if row[j] == 0) for j in range(n)]
    return row_zeros, col_zeros

def transform_matrix(new_matrix, n):
    new_matrix_copy = copy.deepcopy(new_matrix)
    while True:
        row_zeros, col_zeros = count_zeros(new_matrix, n)
        print(f"Row Zeros: {row_zeros}")
        print(f"Column Zeros: {col_zeros}")
        max_row_zeros = max(row_zeros)
        max_col_zeros = max(col_zeros)
        if max_row_zeros == 0 and max_col_zeros == 0:
            break
        if max_row_zeros >= max_col_zeros:
            max_row = row_zeros.index(max_row_zeros)
            print(f"Setting Row {max_row} to 'a' or infinity.")
            new_matrix[max_row] = [
                "a" if value == float('inf') else float('inf') for value in new_matrix[max_row]
            ]
        else:
            max_col = col_zeros.index(max_col_zeros)
            print(f"Setting Column {max_col} to 'a' or infinity.")
            for row in range(n):
                new_matrix[row][max_col] = (
                    "a" if new_matrix[row][max_col] == float('inf') else float('inf')
                )
        print("\n[Output] Matrix after transformation:")
        for row in new_matrix:
            print("\t" + " ".join(str(value) for value in row))
    print("\n[Final] Matrix After All Transformations:")
    for row in new_matrix:
        print("\t" + " ".join(str(value) for value in row))
    return new_matrix, new_matrix_copy

def subtract_min_value(new_matrix, new_matrix_copy, n):
    min_value = min(
        value for row in new_matrix for value in row
        if value != "a" and value != float('inf')
    )
    print(f"Min Value: {min_value}")
    for i in range(n):
        for j in range(n):
            if new_matrix[i][j] != "a" and new_matrix[i][j] != float('inf'):
                new_matrix_copy[i][j] -= min_value
            elif new_matrix[i][j] == "a":
                new_matrix_copy[i][j] += min_value
    print("\n[Output] Final Matrix after Subtraction:")
    for row in new_matrix_copy:
        print("\t" + " ".join(str(value) for value in row))
    return new_matrix_copy

def optimal_assignment(matrix, n):
    cost_matrix = np.array(matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)    
    print("\n[Output] Selected Indices:")
    print(col_ind)
    print("\n[Output] Selected Costs:")
    for i in range(n):
        print(f"{cost_matrix[i][col_ind[i]]}", end=' + ' if i < n - 1 else ' = ')
    total_cost = sum(cost_matrix[i][col_ind[i]] for i in range(n))
    print(f"{total_cost}")

def solve_assignment_problem(n, Arr):
    Arr = preprocess_array(Arr)
    cost_matrix = create_cost_matrix(n, Arr)
    reduced_matrix = reduce_matrix(cost_matrix, n)
    transformed_matrix = transform_matrix(reduced_matrix, n)
    optimal_assignment(cost_matrix, n)
    return transformed_matrix