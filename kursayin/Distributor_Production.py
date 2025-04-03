
import sys
import numpy as np
from tabulate import tabulate
import pandas as pd
import subprocess

class TripleDict:
    def __init__(self, header=None):
        self._store = []
        self.header = header if header else ["Key", "Value 1", "Value 2"]

    def add(self, key, value1, value2):
        """Adds a new entry, allowing duplicate keys."""
        self._store.append((key, value1, value2))

    def items(self):
        """Returns the stored data as a list of tuples."""
        return self._store

    def print_as_table(self):
        """Prints the stored data in a table format."""
        # print(tabulate(self._store, headers=self.header, tablefmt="fancy_grid"))
        # print data coma separated
        print("\n".join([f"{key}, {value1}, {value2}" for key, value1, value2 in self._store]))

    def __repr__(self):
        """Represents the TripleDict contents."""
        return f"TripleDict({self._store})"


def compute_revenue_values(R, alpha, n, C):
    revenue_values = np.zeros((n, C + 1))
    for i in range(n):
        for x in range(1, C + 1):
            revenue_values[i, x] = R[i] * (1-(1 - np.exp(-alpha[i] / x)) ** x)
    return revenue_values

def print_table(data, headers, title):
    """Print table with proper formatting."""
    print(title)
    print(tabulate(data, headers=headers, tablefmt=TABLE_FORMAT, floatfmt=FLOAT_FORMAT))

def solve_resource_allocation(R, alpha, C, n):
    revenue_values = compute_revenue_values(R, alpha, n, C)
    dp = np.zeros((n + 1, C + 1))
    allocation = np.zeros((n + 1, C + 1), dtype=int)

    revenue_table = [[f"i={i + 1}"] + list(revenue_values[i, :]) for i in range(n)]
    headers = ["Prod\\Res"] + [f"c={j}" for j in range(C + 1)]
    print_table(revenue_table, headers, "Revenue Table:")

    transpose = revenue_values.transpose()
    header = ["Res\\Prod"] + [f"i={i}" for i in range(n)]
    print_table(transpose, header, "Transpose:")

    def save_table_to_excel(data, headers, title, filename="output.xlsx", sheet_name="Sheet1"):
        """Save table to an Excel file with proper formatting."""
        df = pd.DataFrame(data, columns=headers)

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Formatting the title
            worksheet.write(0, 0, title)

        print(f"Table saved to {filename}")

    # Example usage
    transposed_revenue = np.transpose(revenue_values)
    rev_table = [[f"c={j}"] + list(transposed_revenue[j, :]) for j in range(C + 1)]
    headers_1 = ["Res\\Prod"] + [f"i={i + 1}" for i in range(n)]
    save_table_to_excel(rev_table, headers_1, "Revenue Table:", "revenue_table.xlsx")

    data = []
    latex_content = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}
"""

    for i in range(1, n + 1):
        dt = TripleDict([f"Z={i}", f"varphi_{i}(z_{i})", f"X_{i}"])  # Updated header to use varphi
        # print(f"\nProcessing product {i}...")
        latex_content += f"\\section{{Product {i}}}\n"
        latex_content += r"\begin{align*}" + "\n"
        latex_content += ("\\n")

        for c in range(C + 1):
            dp[i, c] = dp[i - 1, c]  # Default initialization
            if c > 0:
                # Add empty line for clarity
                latex_content += ("  \n")
                latex_content += ("  \n")
                values = [(dp[i - 1, c - x] + revenue_values[i - 1][x], x) for x in range(1, c + 1)]
                if len(values) > 10:
                    first_values = [f"{float(value)}" for value, x in values[:5]] +['..............'] + [f"{float(value)}" for value, x in values[-5:]]
                else:
                    first_values = [f"{float(value)}" for value, x in values]
                dp[i, c], allocation[i, c] = max(values, key=lambda pair: pair[0])

                # Generate LaTeX for each calculation in the desired format
                calc_str = " \\\\\n ".join([f"{val}" for val in first_values])
                latex_content += (f"\\varphi_{{{i}}}({c}) &= \\max \\left\\{{ \\begin{{array}}{{c}}\n"
                                  f"{calc_str}\n"
                                  f"\\end{{array}} \\right\\}} = {dp[i, c]}, \\quad x_{{{i}}}^0 = {allocation[i, c]}\\\\\n")
                # Add empty line for clarity
                latex_content += ("  \n")
                latex_content += ("  \n")
                # print(f"  z = ({c}) | varphi({c})={dp[i, c]} | X={allocation[i, c]}")
            dt.add(c, dp[i, c], allocation[i, c])

        latex_content += r"\end{align*}" + "\n"

        data.append(dt)

    latex_content += r"""
\end{document}
"""

    # Write LaTeX to file
    with open("Distributor_calculations.tex", "w") as f:
        f.write(latex_content)

    # Print DP table for verification
    dp_table = [[f"i={i}"] + list(dp[i, :]) for i in range(1, n + 1)]
    print_table(dp_table, headers, "DP Table:")

    # Print allocation table
    allocation_table = [[f"i={i}"] + list(row[:]) for i, row in enumerate(allocation) if i > 0]
    print_table(allocation_table, headers, "Allocation Table:")

    # Backtrack optimal allocation
    optimal_allocation = np.zeros(n, dtype=int)
    remaining_resources = C
    for i in range(n, 0, -1):
        optimal_allocation[i - 1] = allocation[i, remaining_resources]
        remaining_resources -= optimal_allocation[i - 1]

    # Compute final revenue
    optimal_revenue = dp[n, C]
    return optimal_allocation, optimal_revenue, data

def print_results(optimal_allocation, R, alpha, C, max_revenue):
    """Print the optimal allocation and resultant revenue."""
    print("\nOptimal Resource Allocation:")
    for i, x in enumerate(optimal_allocation):
        revenue = R[i] * (1 - np.exp(-alpha[i] * x)) ** x
        print(f"Customer {i + 1}: {x} resources -> Revenue: {revenue:.2f}")
    print(f"Total resources used: {sum(optimal_allocation)} out of {C}")
    print(f"Maximum revenue: {max_revenue:.2f}")

# Main Execution
R = np.array([5, 4, 10, 8, 3, 7, 2, 8, 3])
alpha = np.array([3, 4, 5.5, 3, 4.5, 2.5, 4, 5, 4.5])
C = 36
n = 9

# Constants
OUTPUT_FILE_PATH = 'Distributor_results.txt'
OUTPUT_ENCODING = 'utf-8'
TABLE_FORMAT = "fancy_grid"
FLOAT_FORMAT = ".2f"

with open(OUTPUT_FILE_PATH, 'w', encoding=OUTPUT_ENCODING) as output_file:
    sys.stdout = output_file
    optimal_allocation, max_revenue, data = solve_resource_allocation(R, alpha, C, n)
    print_results(optimal_allocation, R, alpha, C, max_revenue)
    sys.stdout = sys.__stdout__

with open("Distributor_tables.txt", 'w', encoding=OUTPUT_ENCODING) as output_file:
    sys.stdout = output_file
    for dt in data:
        dt.print_as_table()
    sys.stdout = sys.__stdout__
