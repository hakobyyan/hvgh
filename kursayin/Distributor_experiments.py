import sys
import numpy as np
from tabulate import tabulate
import pandas as pd

# Constants
OUTPUT_FILE_PATH = 'Distributor_results.txt'
TABLES_FILE_PATH = 'Distributor_tables.txt'
OUTPUT_ENCODING = 'utf-8'
TABLE_FORMAT = "fancy_grid"
FLOAT_FORMAT = ".2f"


class TripleDict:
    """A class to store triples of data and display them as tables."""
    def __init__(self, header=None):
        """Initialize with an optional header; defaults to generic labels."""
        self._store = []
        self.header = header if header else ["Key", "Value 1", "Value 2"]

    def add(self, key, value1, value2):
        """Add an entry with a key and two values, allowing duplicate keys."""
        self._store.append((key, value1, value2))

    def items(self):
        """Return stored data as a list of tuples."""
        return self._store

    def print_as_table(self):
        """Print stored data in a formatted table."""
        print(tabulate(self._store, headers=self.header, tablefmt=TABLE_FORMAT))

    def __repr__(self):
        """Return a string representation of the TripleDict contents."""
        return f"TripleDict({self._store})"


def compute_revenue_values(R, alpha, n, C):
    """Compute revenue for each product and resource amount from 0 to C.

    Args:
        R (np.ndarray): Maximum revenue potential for each product.
        alpha (np.ndarray): Decay parameter for each product.
        n (int): Number of products.
        C (int): Total resources.

    Returns:
        np.ndarray: Revenue values [product, resources], with revenue=0 for x=0.
    """
    revenue_values = np.zeros((n, C + 1))
    for i in range(n):
        for x in range(1, C + 1):
            revenue_values[i, x] = R[i] * (1 - (1 - np.exp(-alpha[i] / x)) ** x)
    return revenue_values


def print_table(data, headers, title):
    """Print a table with the specified data and headers.

    Args:
        data (list): Table data as a list of rows.
        headers (list): Column headers.
        title (str): Title to display above the table.
    """
    print(title)
    print(tabulate(data, headers=headers, tablefmt=TABLE_FORMAT, floatfmt=FLOAT_FORMAT))


def save_table_to_excel(data, headers, title, filename="output.xlsx", sheet_name="Sheet1"):
    """Save a table to an Excel file.

    Args:
        data (list): Table data as a list of rows.
        headers (list): Column headers.
        title (str): Title to include in the Excel file.
        filename (str): Output file name (default: 'output.xlsx').
        sheet_name (str): Excel sheet name (default: 'Sheet1').
    """
    df = pd.DataFrame(data, columns=headers)
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        worksheet.write(0, 0, title)
    print(f"Table saved to {filename}")


def generate_latex(revenue_values, dp, allocation, n, C):
    """Generate a LaTeX document for DP calculations.

    Args:
        revenue_values (np.ndarray): Precomputed revenue values.
        dp (np.ndarray): Dynamic programming table.
        allocation (np.ndarray): Allocation table.
        n (int): Number of products.
        C (int): Total resources.

    Returns:
        str: LaTeX content as a string.
    """
    latex_content = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}
"""
    for i in range(1, n + 1):
        latex_content += f"\\section{{Product {i}}}\n"
        latex_content += r"\begin{align*}" + "\n"
        c_range = [C] if i == n else range(C + 1)
        for c in c_range:
            str_values = [f"{np.round(revenue_values[i - 1, x], 4)} + {np.round(dp[i - 1, c - x], 4)}"
                          for x in range(c + 1)]
            calc_str = " \\\\\n ".join(str_values)
            latex_content += (f"\\varphi_{{{i}}}({c}) &= \\max \\left\\{{ \\begin{{array}}{{c}}\n"
                              f"{calc_str}\n"
                              f"\\end{{array}} \\right\\}}={dp[i, c]}, "
                              f"\\quad x_{{{i}}}^0={allocation[i, c]}\\\\\n")
            if i != n:
                latex_content += "  \n"
        latex_content += " \\\\ \n"  # Spacing for readability
        latex_content += r"\end{align*}" + "\n"
    latex_content += r"\end{document}"
    return latex_content


def fill_dp_table(revenue_values, n, C):
    """Fill DP and allocation tables for optimal resource allocation.
    Args:
        revenue_values (np.ndarray): Precomputed revenue values.
        n (int): Number of products.
        C (int): Total resources.
    Returns:
        tuple: (dp table, allocation table) with maximum revenues and allocations.
    """
    dp = np.zeros((n + 1, C + 1))
    allocation = np.zeros((n + 1, C + 1), dtype=int)
    for i in range(1, n + 1):
        c_range = [C] if i == n else range(C + 1)
        for c in c_range:
            dp[i, c] = dp[i - 1, c]  # Default initialization
            values = [(round(revenue_values[i - 1, x] + dp[i - 1, c - x], 4), x) for x in range(c + 1)]
            dp[i, c], allocation[i, c] = max(values, key=lambda pair: pair[0])
    return dp, allocation

def solve_resource_allocation(R, alpha, C, n):
    """Solve the resource allocation problem using dynamic programming.
    Args:
        R (np.ndarray): Maximum revenues for each product.
        alpha (np.ndarray): Decay parameters for each product.
        C (int): Total resources.
        n (int): Number of products.
    Returns:
        tuple: (optimal_allocation, optimal_revenue, data)
    """
    # Compute revenue values
    revenue_values = compute_revenue_values(R, alpha, n, C)

    # Print and save revenue table
    headers = ["Prod\\Res"] + [f"c={j}" for j in range(C + 1)]
    revenue_table = [[f"i={i + 1}"] + list(revenue_values[i, :]) for i in range(n)]
    print_table(revenue_table, headers, "Revenue Table:")

    # Save transposed revenue table to Excel
    transposed_revenue = np.transpose(revenue_values)
    rev_table = [[f"c={j}"] + list(transposed_revenue[j, :]) for j in range(C + 1)]
    headers_transposed = ["Res\\Prod"] + [f"i={i + 1}" for i in range(n)]
    save_table_to_excel(rev_table, headers_transposed, "Revenue Table:", "revenue_table.xlsx")

    # Fill DP and allocation tables
    dp, allocation = fill_dp_table(revenue_values, n, C)

    # Collect data for tables and LaTeX
    data = []
    for i in range(1, n + 1):
        dt = TripleDict([f"Z={i}", f"varphi_{i}(z_{i})", f"X_{i}"])
        c_range = [C] if i == n else range(C + 1)
        for c in c_range:
            dt.add(c, dp[i, c], allocation[i, c])
        data.append(dt)

    # Generate and save LaTeX document
    latex_content = generate_latex(revenue_values, dp, allocation, n, C)
    with open("Distributor_calculations.tex", "w") as f:
        f.write(latex_content)

    # Print DP and allocation tables
    dp_table = [[f"i={i}"] + list(dp[i, :]) for i in range(1, n + 1)]
    allocation_table = [[f"i={i}"] + list(allocation[i, :]) for i in range(1, n + 1)]
    print_table(dp_table, headers, "DP Table:")
    print_table(allocation_table, headers, "Allocation Table:")

    # Backtrack to find optimal allocation
    optimal_allocation = np.zeros(n, dtype=int)
    remaining_resources = C
    for i in range(n, 0, -1):
        optimal_allocation[i - 1] = allocation[i, remaining_resources]
        remaining_resources -= optimal_allocation[i - 1]

    optimal_revenue = dp[n, C]
    return optimal_allocation, optimal_revenue, data


def print_results(optimal_allocation, R, alpha, C, max_revenue):
    """Print the optimal allocation and corresponding revenues.
    Args:
        optimal_allocation (np.ndarray): Optimal resource allocation per product.
        R (np.ndarray): Maximum revenues for each product.
        alpha (np.ndarray): Decay parameters for each product.
        C (int): Total resources.
        max_revenue (float): Maximum achievable revenue.
    """
    print("\nOptimal Resource Allocation:")
    for i, x in enumerate(optimal_allocation):
        revenue = R[i] * (1 - (1 - np.exp(-alpha[i] / x)) ** x) if x > 0 else 0
        print(f"Customer {i + 1}: {x} resources -> Revenue: {revenue:.2f}")
    print(f"Total resources used: {sum(optimal_allocation)} out of {C}")
    print(f"Maximum revenue: {max_revenue:.2f}")

# Main Execution
if __name__ == "__main__":
    R = np.array([5, 4, 10, 8, 3, 7, 2, 8, 3])
    alpha = np.array([3, 4, 5.5, 3, 4.5, 2.5, 4, 5, 4.5])
    C = 36
    n = 9

    # Redirect output to files
    with open(OUTPUT_FILE_PATH, 'w', encoding=OUTPUT_ENCODING) as output_file:
        sys.stdout = output_file
        optimal_allocation, max_revenue, data = solve_resource_allocation(R, alpha, C, n)
        print_results(optimal_allocation, R, alpha, C, max_revenue)
        sys.stdout = sys.__stdout__

    with open(TABLES_FILE_PATH, 'w', encoding=OUTPUT_ENCODING) as output_file:
        sys.stdout = output_file
        for dt in data:
            dt.print_as_table()
        sys.stdout = sys.__stdout__