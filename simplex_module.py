from operator import eq
from tabulate import tabulate
import pandas as pd
import numpy as np

def check_equations(c, A, b, equalities, max_problem):
    """
    Checks if the initial equations are in the correct form for the simplex method.
    
    Parameters
    ----------
    c : array_like
        The coefficients of the objective function.
    A : array_like
        The coefficients of the constraints.
    b : array_like
        The right hand side of the constraints.
    equalities : list
        The type of each constraint.
    max_problem : bool
        If True, the problem is a maximization problem. Otherwise, it is a minimization problem.
    
    Returns
    -------
    c : array_like
        The coefficients of the objective function, possibly modified.
    A : array_like
        The coefficients of the constraints, possibly modified.
    b : array_like
        The right hand side of the constraints, unchanged.
    equalities : list
        The type of each constraint, possibly modified.
    """
    num_constraints, _ = A.shape
    tableau = np.hstack([b.reshape(-1, 1), A])
    target_columns = np.eye(num_constraints)
    equals = True
    for col in target_columns.T:
        for i in range(1, tableau.shape[1]):
            if np.array_equal(col, tableau[:, i]) and c[i - 1] == 0:
                equals = True
                break
            else:
                equals = False
    if equals:
        print("Initial equations are correct")
        return c, A, b, equalities

    c = list(c)
    A = A.tolist()
    updated_equalities = []

    if max_problem:
        for index, eq in enumerate(equalities):
            if eq == '<=':
                c.append(0)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities.append('=')
            elif eq == '>=':
                c.append(0)
                for row in A[:index]:
                    row.append(0)
                A[index].append(-1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities.append('<=')
            elif eq == '=':
                updated_equalities.append('<=')
        for index, eq in enumerate(updated_equalities):
            if eq == '<=':
                c.append(-np.inf)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities[index] = '='
            elif eq == '>=':
                c.append(-np.inf)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities[index] = '='
            elif eq == '=':
                updated_equalities[index] = '='
    else:
        for index, eq in enumerate(equalities):
            if eq == '<=':
                c.append(0)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities.append('=')
            elif eq == '>=':
                c.append(0)
                for row in A[:index]:
                    row.append(0)
                A[index].append(-1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities.append('<=')
            elif eq == '=':
                updated_equalities.append('<=')
        for index, eq in enumerate(updated_equalities):
            if eq == '<=':
                c.append(np.inf)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities[index] = '='
            elif eq == '>=':
                c.append(np.inf)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities[index] = '='
            elif eq == '=':
                updated_equalities[index] = '='

    tableau = np.hstack([b.reshape(-1, 1), A])
    for col in target_columns.T:
        found = []
        for i in range(1, tableau.shape[1]):
            if np.array_equal(col, tableau[:, i]):
                if c[i - 1] == 0 or c[i - 1] == -np.inf or c[i - 1] == np.inf:
                    found.append(i)
        if len(found) > 1:
            for index, i in enumerate(found):
                if c[i - 1] == -np.inf or c[i - 1] == np.inf:
                    c.pop(i - 1)
                    for row in A:
                        row.pop(i - 1)
                    break

    return np.array(c), np.array(A), b, updated_equalities


def prime_to_dual(c, A, b, is_minimization, equalities):
    """
    Converts a primal problem into its dual.

    Parameters
    ----------
    c : array_like
        Coefficients of the linear objective function.
    A : array_like
        Coefficient matrix of the constraints.
    b : array_like
        Right-hand side of the constraints.
    is_minimization : bool
        Whether the primal problem is a minimization problem.
    equalities : list
        List of equality types, '<=', '>=', or '='.

    Returns
    -------
    dual_c : array_like
        Coefficients of the linear objective function of the dual problem.
    dual_A : array_like
        Coefficient matrix of the constraints of the dual problem.
    dual_b : array_like
        Right-hand side of the constraints of the dual problem.
    is_dual_maximization : bool
        Whether the dual problem is a maximization problem.
    dual_equalities : list
        List of equality types of the dual problem.
    """
    AA = A
    bb = b
    eq_1 = equalities
    dual_A = AA.T
    dual_b = c
    dual_c = bb
    dual_equalities = []
    indexes = []
    for index, row in enumerate(dual_A):
        if not is_minimization:
            dual_equalities.append('<=')
        else:
            dual_equalities.append('>=')

    while '=' in eq_1:
        for index, eq in enumerate(eq_1):
            if eq == '=':
                eq_1[index] = '>='
                eq_1 = np.insert(eq_1, index+1, '>=',)
                column = -dual_A[:, index]
                dual_A = np.insert(dual_A, index+1, column, axis=1)
                dual_c = np.insert(dual_c, index+1, -dual_c[index])
                break
    return dual_c, dual_A, dual_b, not is_minimization, dual_equalities


def print_tableau(tableau):
    """
    Print the tableau of the Linear Programming problem in a more readable format.

    Parameters:
    tableau (np.ndarray): The tableau of the Linear Programming problem.
    """
    columns = ['Basis', 'C_basis', 'Solution'] + [f'x{i}' for i in range(1, tableau.shape[1] - 2)]
    index = [f'Row{i}' for i in range(1, tableau.shape[0])]
    index.append('Objective')

    df = pd.DataFrame(tableau, columns=columns, index=index)
    table_str = tabulate(df, headers="keys", tablefmt="grid", numalign="center", stralign="center")
    print(table_str)


def print_solution(tableau, c=None):
    """
    Print the solution of the Linear Programming problem.

    Parameters:
    tableau (np.ndarray): The final tableau of the Linear Programming problem.
    c (int): The number of variables in the original problem, if given, the function prints the objective function of the original problem.
    """
    basis = tableau[:, 0][:-1]
    solution = tableau[:, 2][:-1]
    objective_function = []
    for i in range(tableau.shape[1] - 3):
        if i + 1 not in basis:
            objective_function.append(0)
        else:
            index = np.where(basis == i + 1)[0][0]
            objective_function.append(solution[index])
    if c is not None:
        print(f"Final objective Function: {objective_function[:c]}")
    else:
        print(f"Objective Function: {objective_function}")


def create_inequalities(c, a, b, equalities, max_problem):
    """
    Creates a string representation of the linear programming problem.

    Parameters
    ----------
    c : array_like
        The coefficients of the objective function.
    a : array_like
        The coefficients of the constraints.
    b : array_like
        The right-hand side of the constraints.
    equalities : list
        A list of strings indicating the type of each constraint. Each string
        can be '<=', '>=', or '='.
    max_problem : bool
        If True, the problem is a maximization problem.

    Returns
    -------
    str
        A string representation of the linear programming problem.
    """
    objective = "max" if max_problem else "min"
    objective_terms = [f"{c[i]}*x{i + 1}" for i in range(len(c))]
    objective_str = f"{objective} Z = " + " + ".join(objective_terms)

    constraints = []
    for i in range(len(a)):
        terms = [f"{a[i, j]}*x{j + 1}" for j in range(a.shape[1])]
        constraint = " + ".join(terms) + f" {equalities[i]} {b[i]}"
        constraints.append(constraint)

    non_negativity = [f"x{i + 1} >= 0" for i in range(len(c))]

    result = objective_str + "\nRestrictions:\n"
    result += "\n".join(constraints) + "\n"
    result += ", ".join(non_negativity)
    result += "\n"
    return result

def calc_tetthal(c_row, table, max_problem, basis):
    """
    Calculate the entering and leaving variables for the current iteration 
    of the simplex method.

    Parameters:
    c_row (np.ndarray): The cost row vector from the current tableau.
    table (np.ndarray): The current simplex tableau.
    max_problem (bool): Indicates if the problem is a maximization problem.
    basis (np.ndarray): The current basis indices.

    Returns:
    tuple: A tuple containing the index of the entering variable and 
           the index of the leaving variable. Returns (None, None) if 
           no valid pivot can be found.
    """
    tettha = []
    solution_column = table[:, 2][:len(table[:, 2]) - 1]
    if max_problem:
        values = np.where((c_row < 0) & (c_row != -np.inf) & (c_row != 0))[0]
        min_val = np.min(c_row[values])
    else:
        values = np.where((c_row > 0) & (c_row != np.inf) & (c_row != 0))[0]
        min_val = np.max(c_row[values])
    values = values[c_row[values] == min_val]
    for index in values:
        positive_ratios = np.where(table[:, index + 3][:len(table[:, 2]) - 1] >= 0, solution_column / table[:, index + 3][:len(table[:, 2]) - 1], np.inf)
        positive_ratios = np.where(positive_ratios >= 0, positive_ratios, np.inf)
        if len(positive_ratios) > 0:
            min_ratio = np.min(positive_ratios)
            min_indices = np.where(positive_ratios == min_ratio)[0]
            latest_index = min_indices[0]
            tettha.append([index, latest_index])
    tettha = np.array(tettha)
    selected_index = np.argmin(tettha[:, 1])
    index = tettha[selected_index][0]
    if all(table[:-1, index+3] <= 0):
        return None, None
    remove_index = tettha[selected_index][1]
    if any(table[:, 1] == np.inf):
        remove_index = np.where(table[:, 1] == np.inf)[0]
        remove_index = remove_index[0] if len(remove_index) > 0 else None
    return int(index) + 1, int(remove_index) if remove_index is not None else None


def simplex_method(c, a, b, equalities, max_problem, prime=False):
    print("Initial Problem:")
    print(create_inequalities(c, a, b, equalities, max_problem))
    if prime:
        print("Converting to dual problem")
        c, a, b, max_problem, equalities = prime_to_dual(c, a, b, max_problem, equalities)
        print(create_inequalities(c, a, b, equalities, max_problem))
        
    for index, Val in enumerate(b):
        if Val < 0:
            b[index] = -Val
            a[index] = -a[index]
            equalities[index] = '<='
    print(create_inequalities(c, a, b, equalities, max_problem))
    c, a, b, equalities = check_equations(c, a, b, equalities, max_problem)
    len_C = c.shape[0]
    print("Canonical Problem:")
    print(create_inequalities(c, a, b, equalities, max_problem))

    num_constraints, num_variables = a.shape
    tableau = np.hstack([b.reshape(-1, 1), a])
    C_basis = np.zeros(num_constraints)
    target_columns = np.eye(num_constraints)
    c_row = np.dot(C_basis, a) - c
    c_row = np.hstack([0, c_row])

    artificial = False
    for col in target_columns.T:
        if not any(np.array_equal(col, tableau[:, i])  for i in range(1, tableau.shape[1])):
            c_row = np.hstack([c_row, 0])

    tableau = np.vstack([tableau, c_row])
    basis = []
    for col_index, target_col in enumerate(target_columns.T):
        for tableau_col_index in range(1, tableau.shape[1]):
            if np.allclose(tableau[:-1, tableau_col_index], target_col, atol=1e-5):
                if c_row[tableau_col_index] == 0 or c_row[tableau_col_index] == -np.inf or c_row[tableau_col_index] == np.inf:
                    basis.append(tableau_col_index)
                    break

    if len(basis) < num_constraints:
        for i in range(num_constraints):
            if i + 1 not in basis:
                basis.append(i + 1)
    basis = np.array(basis[:num_constraints], dtype=int)

    for index, i in enumerate(basis[:-1]):
        if -float(c_row[int(i)]) == -0:
            C_basis[index] = float(c_row[int(i)])
        else:
            C_basis[index] = -float(c_row[int(i)])

    c_row_original = c_row
    c_basis = np.hstack([C_basis, np.zeros(num_constraints)])[:tableau.shape[0]]
    c_basis = np.zeros(tableau.shape[0] - 1)
    for i, col_index in enumerate(basis):
        if col_index <= len(c):
            c_basis[i] = c[col_index - 1]
        else:
            c_basis[i] = 0

    if -np.inf in c or np.inf in c:
        M = 10000000
        c_M = [M if i == np.inf else -M if i == -np.inf else i for i in c]
        c_M = np.hstack([c_M, np.zeros(num_constraints)])[:tableau.shape[1] - 1]
        C_basis_M = [M if i == np.inf else -M if i == -np.inf else i for i in c_basis]
        c_row = (C_basis_M @ tableau[:-1, 1:]) - c_M
        artificial = True

    tableau[-1, 0] = np.dot(C_basis, tableau[:-1, 0])
    for i in range(1, len(tableau[:-1, 1:])):
        if artificial:
            tableau[-1, i] = np.dot(C_basis, tableau[:-1, i]) + c_row_original[i]
        else:
            tableau[-1, i] = np.dot(C_basis, tableau[:-1, i]) + c_row[i]
    c_row = tableau[-1, :]

    basis_full = np.zeros(tableau.shape[0] - 1, dtype=int)
    basis_full[:len(basis)] = basis

    c_basis_full = np.zeros(tableau.shape[0] - 1)
    c_basis_full[:len(c_basis)] = c_basis

    tableau = np.hstack([
        np.append(basis_full, [0]).reshape(-1, 1),
        np.append(c_basis_full, [0]).reshape(-1, 1),
        tableau])
    if artificial:
        tableau = np.where(tableau == np.inf, M, tableau)
        tableau = np.where(tableau == -np.inf, -M, tableau)
        c_original_M = [M if i == np.inf else -M if i == -np.inf else i for i in c_row_original]
        tableau[-1, 2] = np.dot(tableau[:-1, 1], tableau[:-1, 2])
        for i in range(3, 3 + tableau[:-1, 3:].shape[1]):
            tableau[-1, i] = (tableau[:-1, 1] @ tableau[:-1, i]) + c_original_M[i - 2]
        c_row = tableau[-1, 2:]

    print_solution(tableau)
    print_tableau(tableau)

    

    if max_problem:
        while any(c_row[1:] < 0):
            negatives = [x for x in c_row[1:] if x < 0]
            if all(x == -np.inf for x in negatives):
                break
            print(f"Negatives: {negatives}")
            print("Initial tableau is not optimal")
            new_basis, remove_basis = calc_tetthal(c_row[1:], tableau, max_problem, basis)
            if new_basis is None:
                print("Problem is unbounded")
                return
            print(f"New basis is {new_basis}")
            remove_basis = basis[remove_basis]
            print(f"Remove basis is {int(remove_basis)}")
            line = np.where(basis == remove_basis)[0][0]
            tableau[line, 0] = new_basis
            tableau[line, 1] = -c_row_original[new_basis] if c_row_original[new_basis] != 0 else c_row_original[new_basis]
            tableau[line, 2:] = np.round(tableau[line][2:] / tableau[line][3 + new_basis - 1], 2)
            tableau[:line, 2:] = np.round(tableau[:line, 2:] - tableau[:line, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 2)
            tableau[line + 1:, 2:] = np.round(tableau[line + 1:, 2:] - tableau[line + 1:, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 2)
            tableau[-1, 2] = np.dot(tableau[:-1, 1], tableau[:-1, 2])
            if artificial:
                for i in range(3, 3 + tableau[:-1, 3:].shape[1]):
                    tableau[-1, i] = (tableau[:-1, 1] @ tableau[:-1, i]) + c_original_M[i - 2]
            else:
                for i in range(3, len(tableau[:-1, 3:])):
                    tableau[-1, i] = np.dot(tableau[:-1, 1], tableau[:-1, i]) + c_row_original[i - 2]
            c_row = tableau[-1, 2:]
            basis[line] = new_basis
            print_solution(tableau)
            print_tableau(tableau)
        if -10000000 in tableau[:, 1]:
            print("Problem has no feasible solution")
            return
        else:
            print("Initial tableau is optimal")
    else:
        while any(c_row[1:] > 0):
            positives = [x for x in c_row[1:] if x > 0]
            if all(x == -np.inf for x in positives):
                break
            print(f"Positives: {positives}")
            print("Initial tableau is not optimal")
            new_basis, remove_basis = calc_tetthal(c_row[1:], tableau, max_problem, basis)
            if new_basis is None:
                print("Problem is unbounded")
                return
            print(f"New basis is {new_basis}")
            remove_basis = basis[remove_basis]
            print(f"Remove basis is {int(remove_basis)}")
            line = np.where(basis == remove_basis)[0][0]
            tableau[line, 0] = new_basis
            tableau[line, 1] = -c_row_original[new_basis] if c_row_original[new_basis] != 0 else c_row_original[new_basis]
            tableau[line, 2:] = np.round(tableau[line][2:] / tableau[line][3 + new_basis - 1], 2)
            tableau[:line, 2:] = np.round(tableau[:line, 2:] - tableau[:line, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 2)
            tableau[line + 1:, 2:] = np.round(tableau[line + 1:, 2:] - tableau[line + 1:, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 2)
            tableau[-1, 2] = np.dot(tableau[:-1, 1], tableau[:-1, 2])
            if artificial:
                for i in range(3, 3 + tableau[:-1, 3:].shape[1]):
                    tableau[-1, i] = (tableau[:-1, 1] @ tableau[:-1, i]) + c_original_M[i - 2]
            else:
                for i in range(3, len(tableau[:-1, 3:])):
                    tableau[-1, i] = np.dot(tableau[:-1, 1], tableau[:-1, i]) + c_row_original[i - 2]
            c_row = tableau[-1, 2:]
            basis[line] = new_basis
            print_solution(tableau)
            print_tableau(tableau)
        if 10000000 in tableau[:, 1]:
            print("Problem has no feasible solution")
            return
        else:
            print("Initial tableau is optimal")
    print_solution(tableau, len_C)
    print("Value of the objective function: ", tableau[-1, 2])
