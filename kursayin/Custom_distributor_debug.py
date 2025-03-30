import sys
import numpy as np


class TripleDict:
    def __init__(self, header=None):
        self._store = []
        self.header = header if header else ["Key", "Value 1", "Value 2"]

    def add(self, key, value1, value2):
        self._store.append((key, value1, value2))

    def print_as_table(self):
        column_widths = [max(len(str(cell)) for cell in col) for col in zip(*([self.header] + self._store))]

        def format_row(row):
            return " | ".join(f"{str(cell):<{column_widths[i]}}" for i, cell in enumerate(row))

        print(format_row(self.header))
        print("-+-".join("-" * width for width in column_widths))
        for row in self._store:
            print(format_row(row))


OUTPUT_FILE_PATH = 'with_0.txt'
OUTPUT_ENCODING = 'utf-8'
FLOAT_FORMAT = ".2f"


def compute_revenue_values(R, alpha, n, C):
    revenue_values = np.zeros((n, C + 1))
    for i in range(n):
        for x in range(1, C + 1):
            revenue_values[i, x] = R[i] * (1 - np.exp(-alpha[i] * x)) ** x
    return revenue_values


def print_table(data, headers, title):
    print(title)
    column_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + data))]

    def format_row(row):
        return " | ".join(f"{str(cell):<{column_widths[i]}}" for i, cell in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in column_widths))
    for row in data:
        print(format_row(row))


def solve_resource_allocation(R, alpha, C, n):
    revenue_values = compute_revenue_values(R, alpha, n, C)
    dp = np.zeros((n + 1, C + 1))
    allocation = np.zeros((n + 1, C + 1), dtype=int)

    revenue_table = [[f"i={i + 1}"] + list(revenue_values[i, :]) for i in range(n)]
    headers = ["Prod\\Res"] + [f"c={j}" for j in range(C + 1)]
    print_table(revenue_table, headers, "Revenue Table:")

    data = []
    for i in range(1, n + 1):
        dt = TripleDict([f"Z={i}", f"fi_{i}(z_{i})", f"X_{i}"])
        print(f"\nProcessing product {i}...")
        for c in range(C + 1):
            dp[i, c] = dp[i - 1, c]
            print(f"  Resource count {c}: Initial dp[{i},{c}] = {dp[i, c]}")
            for x in range(c + 1):
                value = dp[i - 1, c - x] + revenue_values[i - 1, x]
                print(f"    Trying allocation c={x}, computed value={value}")
                if value > dp[i, c]:
                    dp[i, c] = value
                    allocation[i, c] = x
                    print(f"    Updated dp[{i},{c}] = {dp[i, c]}, allocation[{i},{c}] = {allocation[i, c]}")
            dt.add(c, dp[i, c], allocation[i, c])
        data.append(dt)

    dp_table = [[f"i={i}"] + list(dp[i, :]) for i in range(1, n + 1)]
    print_table(dp_table, headers, "DP Table:")

    allocation_table = [[f"i={i}"] + list(row[:]) for i, row in enumerate(allocation) if i > 0]
    print_table(allocation_table, headers, "Allocation Table:")

    optimal_allocation = np.zeros(n, dtype=int)
    remaining_resources = C
    for i in range(n, 0, -1):
        optimal_allocation[i - 1] = allocation[i, remaining_resources]
        remaining_resources -= optimal_allocation[i - 1]

    optimal_revenue = dp[n, C]
    return optimal_allocation, optimal_revenue, data


def print_results(optimal_allocation, R, alpha, C, max_revenue):
    print("\nOptimal Resource Allocation:")
    for i, x in enumerate(optimal_allocation):
        revenue = R[i] * (1 - np.exp(-alpha[i] * x)) ** x
        print(f"Customer {i + 1}: {x} resources -> Revenue: {revenue:.2f}")
    print(f"Total resources used: {sum(optimal_allocation)} out of {C}")
    print(f"Maximum revenue: {max_revenue:.2f}")


R = np.array([5, 4, 10, 8, 3, 7, 2, 8, 3])
alpha = np.array([3, 4, 5.5, 3, 4.5, 2.5, 4, 5, 4.5])
C = 36
n = 9

with open(OUTPUT_FILE_PATH, 'w', encoding=OUTPUT_ENCODING) as output_file:
    sys.stdout = output_file
    optimal_allocation, max_revenue, data = solve_resource_allocation(R, alpha, C, n)
    print_results(optimal_allocation, R, alpha, C, max_revenue)
    sys.stdout = sys.__stdout__

with open("tables.txt", 'w', encoding=OUTPUT_ENCODING) as output_file:
    sys.stdout = output_file
    for dt in data:
        dt.print_as_table()
        print("===============================================")
    sys.stdout = sys.__stdout__
