from tabulate import tabulate
import pandas as pd
import numpy as np
import math

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
                c.append(0)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities.append('=')
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
                c.append(0)
                for row in A[:index]:
                    row.append(0)
                A[index].append(1)
                for row in A[index + 1:]:
                    row.append(0)
                updated_equalities.append('=')
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
            i = found[-1]
            if c[i - 1] == 0 or c[i - 1] == -np.inf or c[i - 1] == np.inf:
                c.pop(i - 1)
                for row in A:
                    row.pop(i - 1)
                tableau = np.hstack([b.reshape(-1, 1), A])

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
                eq_1 = np.insert(eq_1, index + 1, '>=', )
                column = -dual_A[:, index]
                dual_A = np.insert(dual_A, index + 1, column, axis=1)
                dual_c = np.insert(dual_c, index + 1, -dual_c[index])
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
    table_str = tabulate(df, headers="keys", tablefmt="fancy_grid", numalign="center", stralign="center")
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
        print(f"Final objective Function: {[round(float(val), 4) for val in objective_function[:c]]}")
    else:
        print(f"Objective Function: {[round(float(val), 4) for val in objective_function]}")
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
        positive_ratios = np.where(table[:, index + 3][:len(table[:, 2]) - 1] >= 0,
                                   solution_column / table[:, index + 3][:len(table[:, 2]) - 1], np.inf)
        positive_ratios = np.where(positive_ratios >= 0, positive_ratios, np.inf)
        if len(positive_ratios) > 0:
            min_ratio = np.min(positive_ratios)
            min_indices = np.where(positive_ratios == min_ratio)[0]
            latest_index = min_indices[0]
            tettha.append([index, latest_index])
    tettha = np.array(tettha)
    selected_index = np.argmin(tettha[:, 1])
    index = tettha[selected_index][0]
    if all(table[:-1, index + 3] <= 0):
        return None, None
    remove_index = tettha[selected_index][1]
    if any(table[:, 1] == np.inf):
        remove_index = np.where(table[:, 1] == np.inf)[0]
        remove_index = remove_index[0] if len(remove_index) > 0 else None
    return int(index) + 1, int(remove_index) if remove_index is not None else None

def calc_tethha_double(solutions, table):
    c_row = table[-1, 3:]
    negatives = np.where(solutions < 0)[0]
    if len(negatives) > 0:
        neg = np.min(solutions[negatives])
        if len(np.where(solutions == neg)[0]) > 1:
            removes, indexes = [], []
            negatives = np.where(solutions == neg)[0]
            for negs in negatives:
                neg_index = np.where(table[negs, 3:] < 0)[0]
                if len(neg_index) > 0:
                    remove, new_index = [np.inf]* len(neg_index), [None] * len(neg_index)
                    for i, index in enumerate(neg_index):
                        val = abs(c_row[index] / table[negs, 3:][index])
                        if val < remove[i]:
                            remove[i] = val
                            new_index[i] = index
                    removes.append(min(remove))
                    indexes.append(np.argmin(remove))
                elif len(neg_index) == 0:
                    print("No negative values in row")
                    return None, None
            new_index = indexes[np.argmax(removes)] if len(removes) > 0 else None
            remove= neg_index[new_index]
        else:
            row = np.where(solutions == neg)[0][0]
            negatives = np.where(table[row, 3:] < 0)[0]
            print(f"Negatives: {negatives}")
            if len(negatives) > 0:
                new, new_index = np.inf, None
                for index, negative in enumerate(negatives):
                    val = abs(c_row[negative] / table[row, 3:][negative])
                    if val < new:
                        new = val
                        new_index = negative

            elif len(negatives) == 0:
                print("No negative values in row")
                return None, None
            else:
                new_index = None
            remove= row
    else:
        remove= None
    return new_index+1, remove

def integer_problem(table, c_row, c_row_original, c_original_M, non_negatives, added_column, remove):
    if added_column and remove:
        # remove last column
        table = table[:, :-1]
        c_row = c_row[:-1]
        c_row_original = c_row_original[:-1]
        c_original_M = c_original_M[:-1]
        added_column = False
    solutions = table[:-1, 2]
    indexes = table[:-1, 0]
    nonNegVals = {}
    for index, x in enumerate(solutions):
        if len(non_negatives) > indexes[index]-1:
            if int(x) != x and non_negatives[int(indexes[index]-1)]:
                nonNegVals[index] = float(x)
    print(f"Non-integer values: {nonNegVals}")
    if nonNegVals == {}:
        print("All values are integers")
        return None, None, None, None, None, None
    if not np.all(list(nonNegVals.values()) == np.floor(list(nonNegVals.values()))):
        print("\033[0;31mProblem is not integer\033[0m")
        doubles, real_doubles = {}, {}
        for i in range(len(nonNegVals)):
            val = list(nonNegVals.values())[i] - math.floor(list(nonNegVals.values())[i])
            doubles[list(nonNegVals.keys())[i]] = (val)
            if val != 0:
                real_doubles[i] = val
        max_double = max(list(doubles.values()))
        max_row = [row for row, value in doubles.items() if value == max_double][0]
        print(f"Selected float: {round(max_double, 3)}\nSelected Row{max_row+1}")
        basis = table[:-1, 0]
        row = [table[max_row][3 + i] - math.floor(table[max_row][3 + i]) for i in
               range(len(table[max_row][3:]))]
        new_row = np.array(-max_double)
        for index, val in enumerate(row):
            if index+1 in basis:
                new_row = np.append(new_row, 0)
            else:
                new_row = np.append(new_row, -row[index])
        new_row = np.insert(new_row, 0, 0)
        new_row = np.insert(new_row, 0, table.shape[1] - 2)
        table = np.insert(table, -1, new_row, axis=0)
        new_column = np.zeros(table.shape[0])
        new_column[-2] = 1
        new_column = new_column.reshape(-1, 1)
        table = np.append(table, new_column, axis=1)
        new_basis  = table[:-1, 0]
        c_row  = table[-1, 2:]
        c_row_original = np.append(c_row_original, 0)
        c_original_M = np.append(c_original_M, 0)
        print_tableau(table)
        added_column = True
        return table, new_basis, c_row, c_row_original, c_original_M, added_column
    else:
        return None

def double_simplex(tableau, c_row_original, c_original_M, artificial, basis, precision=2):
    solution_column = tableau[:, 2][:len(tableau[:, 2]) - 1]

    while any(solution_column < 0):
        negatives = [x for x in solution_column if x < 0]
        print("Initial tableau has negative values in solution column.")
        new_basis, remove_basis = calc_tethha_double(solution_column, tableau)
        if new_basis is None and remove_basis is None:
            return None, None, None, None, None
        remove_basis = basis[remove_basis]
        print(f"New basis is {new_basis}")
        print(f"Remove basis is {remove_basis}")
        line = np.where(basis == remove_basis)[0][0]
        tableau[line, 0] = new_basis
        tableau[line, 1] = -c_row_original[new_basis] if c_row_original[new_basis] != 0 else c_row_original[
            new_basis]
        tableau[line, 2:] = np.round(tableau[line][2:] / tableau[line][3 + new_basis - 1], 4)
        tableau[:line, 2:] = np.round(
            tableau[:line, 2:] - tableau[:line, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 4)
        tableau[line + 1:, 2:] = np.round(
            tableau[line + 1:, 2:] - tableau[line + 1:, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 4)
        tableau[-1, 2] = np.dot(tableau[:-1, 1], tableau[:-1, 2])
        if artificial:
            for i in range(3, 3 + tableau[:-1, 3:].shape[1]):
                tableau[-1, i] = (tableau[:-1, 1] @ tableau[:-1, i]) + c_original_M[i - 2]
        else:
            for i in range(3, len(tableau[:-1, 3:])):
                tableau[-1, i] = np.dot(tableau[:-1, 1], tableau[:-1, i]) + c_row_original[i - 2]
        c_row = tableau[-1, 2:]
        tableau[:, 2][:len(tableau[:, 2]) ] = np.round(tableau[:, 2][:len(tableau[:, 2]) ], precision)
        basis[line] = new_basis
        print_solution(tableau)
        print_tableau(tableau)
    if -10000000 in tableau[:, 1]:
        print("Problem has no feasible solution")
        return
    else:
        print("Negativity is removed")

    return tableau, c_row_original, c_original_M, artificial, basis

def proccess_table(tableau, c_row, c_row_original, c_original_M, artificial, max_problem, basis):
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
                return None
            print(f"New basis is {new_basis}")
            remove_basis = basis[remove_basis]
            print(f"Remove basis is {int(remove_basis)}")
            line = np.where(basis == remove_basis)[0][0]
            tableau[line, 0] = new_basis
            tableau[line, 1] = -c_row_original[new_basis] if c_row_original[new_basis] != 0 else c_row_original[
                new_basis]
            tableau[line, 2:] = np.round(tableau[line][2:] / tableau[line][3 + new_basis - 1], 4)
            tableau[:line, 2:] = np.round(
                tableau[:line, 2:] - tableau[:line, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 4)
            tableau[line + 1:, 2:] = np.round(
                tableau[line + 1:, 2:] - tableau[line + 1:, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 4)
            tableau[-1, 2] = np.dot(tableau[:-1, 1], tableau[:-1, 2])
            if artificial:
                for i in range(3, 3 + tableau[:-1, 3:].shape[1]):
                    tableau[-1, i] = (tableau[:-1, 1] @ tableau[:-1, i]) + c_original_M[i - 2]
            else:
                for i in range(3, len(tableau[:-1, 3:])):
                    tableau[-1, i] = np.dot(tableau[:-1, 1], tableau[:-1, i]) + c_row_original[i - 2]
            c_row = tableau[-1, 2:]
            # tableau[:, 2][:len(tableau[:, 2]) ] = np.round(tableau[:, 2][:len(tableau[:, 2]) ], 2)

            basis[line] = new_basis
            print_solution(tableau)
            print_tableau(tableau)
        if -10000000 in tableau[:, 1]:
            print("Problem has no feasible solution")
            return None
        else:
            print("\033[92mInitial tableau is optimal\033[0m")
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
                return None
            print(f"New basis is {new_basis}")
            remove_basis = basis[remove_basis]
            print(f"Remove basis is {int(remove_basis)}")
            line = np.where(basis == remove_basis)[0][0]
            tableau[line, 0] = new_basis
            tableau[line, 1] = -c_row_original[new_basis] if c_row_original[new_basis] != 0 else c_row_original[
                new_basis]
            tableau[line, 2:] = np.round(tableau[line][2:] / tableau[line][3 + new_basis - 1], 4)
            tableau[:line, 2:] = np.round(
                tableau[:line, 2:] - tableau[:line, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 4)
            tableau[line + 1:, 2:] = np.round(
                tableau[line + 1:, 2:] - tableau[line + 1:, 3 + new_basis - 1].reshape(-1, 1) * tableau[line, 2:], 4)
            tableau[-1, 2] = np.dot(tableau[:-1, 1], tableau[:-1, 2])
            if artificial:
                for i in range(3, 3 + tableau[:-1, 3:].shape[1]):
                    tableau[-1, i] = (tableau[:-1, 1] @ tableau[:-1, i]) + c_original_M[i - 2]
            else:
                for i in range(3, len(tableau[:-1, 3:])):
                    tableau[-1, i] = np.dot(tableau[:-1, 1], tableau[:-1, i]) + c_row_original[i - 2]
            c_row = tableau[-1, 2:]
            # tableau[:, 2][:len(tableau[:, 2])] = np.round(tableau[:, 2][:len(tableau[:, 2])], 2)
            basis[line] = new_basis
            print_solution(tableau)
            print_tableau(tableau)
        if 10000000 in tableau[:, 1]:
            print("Problem has no feasible solution")
            return None
        else:
            print("\033[92mInitial tableau is optimal\033[0m")
    return tableau

def simplex_method(c, a, b, equalities, max_problem, prime=False, double = False, integer = False, precision = 2, non_negatives = None, remove = True):
    print("Initial Problem:")
    print(create_inequalities(c, a, b, equalities, max_problem))
    if prime:
        print("Converting to dual problem")
        c, a, b, max_problem, equalities = prime_to_dual(c, a, b, max_problem, equalities)
        print(create_inequalities(c, a, b, equalities, max_problem))

    # if integer:
    #     for index, Val in enumerate(b):
    #         if equalities[index] == '>=':
    #             print("Flipping")
    #             b[index] = -Val
    #             a[index] = -a[index]
    #             equalities[index] = '<='
    #     double = True
    if not double:
        for index, Val in enumerate(b):
            if Val < 0:
                b[index] = -Val
                a[index] = -a[index]
                if equalities[index] == '<=':
                    equalities[index] = '>='
                elif equalities[index] == '>=':
                    equalities[index] = '<='
    # print(create_inequalities(c, a, b, equalities, max_problem))
    c, a, b, equalities = check_equations(c, a, b, equalities, max_problem)
    len_C = c.shape[0]
    print("Canonical Problem:")
    print(create_inequalities(c, a, b, equalities, max_problem))

    if non_negatives is None:
        non_negatives = [True for i in range(len_C)]
    print(f"non_negatives = {non_negatives}")

    num_constraints, num_variables = a.shape
    tableau = np.hstack([b.reshape(-1, 1), a])
    C_basis = np.zeros(num_constraints)
    target_columns = np.eye(num_constraints)
    c_row = np.dot(C_basis, a) - c
    c_row = np.hstack([0, c_row])

    artificial = False
    for col in target_columns.T:
        if not any(np.array_equal(col, tableau[:, i]) for i in range(1, tableau.shape[1])):
            c_row = np.hstack([c_row, 0])

    tableau = np.vstack([tableau, c_row])
    basis = []
    for col_index, target_col in enumerate(target_columns.T):
        for tableau_col_index in range(1, tableau.shape[1]):
            if np.allclose(tableau[:-1, tableau_col_index], target_col, atol=1e-5):
                if c_row[tableau_col_index] == 0 or c_row[tableau_col_index] == -np.inf or c_row[
                    tableau_col_index] == np.inf:
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
        artificial_indexes = [index+3 for index, value in enumerate(c) if value == np.inf or value == -np.inf]
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
    c_original_M = []
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

    if double:
        tableau, c_row_original, c_original_M, artificial, basis = double_simplex(tableau, c_row_original, c_original_M, artificial, basis, precision)
        if tableau is None:
            return None
        c_row = tableau[-1, 2:]
    tableau = proccess_table(tableau, c_row, c_row_original, c_original_M, artificial, max_problem, basis)
    if tableau is None:
        return None
    print_solution(tableau, len_C)
    print(f"\033[92mValue of the objective function: {tableau[-1, 2]}\033[0m")

    if integer:
        added_column = False
        # Remove artificial variable columns
        if artificial:
            tableau = np.delete(tableau, artificial_indexes, axis=1)
            artificial_indexes = [i - 2 for i in artificial_indexes]
            c_row = np.delete(c_row, artificial_indexes)
            c_row_original = np.delete(c_row_original, artificial_indexes)
            c_original_M = np.delete(c_original_M, artificial_indexes)
            artificial = False

        solutions = tableau[:-1, 2]
        while not np.all(solutions == np.floor(solutions)):
            n_tableau, basis, c_row, c_row_original, c_original_M, added_column= integer_problem(tableau, c_row, c_row_original, c_original_M, non_negatives, added_column, remove)
            if n_tableau is None:
                break
            else:
                tableau = n_tableau
            tableau, c_row_original, c_original_M, artificial, basis = double_simplex(tableau, c_row_original, c_original_M, artificial, basis, precision)
            if tableau is None:
                return None
            tableau= proccess_table(tableau, c_row, c_row_original, c_original_M, artificial, max_problem, basis)
            if tableau is None:
                return None
            solutions = tableau[:-1, 2]
        print_solution(tableau)
        print(f"\033[92mValue of the objective function: {tableau[-1, 2]}\033[0m")
        print("\033[92mInitial solution is integer.\033[0m")
    return tableau
