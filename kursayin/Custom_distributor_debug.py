import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class TripleDict:
    def __init__(self, header=None):
        self._store = []
        self.header = header if header else ["Key", "Value 1", "Value 2"]

    def add(self, key, value1, value2):
        self._store.append((key, value1, value2))

    def items(self):
        return self._store

    def print_as_table(self):
        print("\n".join([f"{key}, {value1}, {value2}" for key, value1, value2 in self._store]))

    def to_dataframe(self):
        return pd.DataFrame(self._store, columns=self.header)

def compute_revenue_values(R, alpha, n, C):
    revenue_values = np.zeros((n, C + 1))
    for i in range(n):
        for x in range(1, C + 1):
            revenue_values[i, x] = R[i] * (1 - (1 - np.exp(-alpha[i] / x)) ** x)
    return revenue_values

def print_table(data, headers, title):
    print(title)
    print(tabulate(data, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))

def plot_allocation(allocation, n, C):
    plt.figure(figsize=(10, 6))
    sns.heatmap(allocation[1:], annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Allocation Table")
    plt.xlabel("Resources (c)")
    plt.ylabel("Product (i)")
    plt.tight_layout()
    plt.show()

def plot_optimal_allocation(optimal_allocation):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=np.arange(1, len(optimal_allocation)+1), y=optimal_allocation, palette="viridis")
    plt.title("Optimal Resource Allocation")
    plt.xlabel("Product")
    plt.ylabel("Allocated Resources")
    plt.tight_layout()
    plt.show()

def solve_resource_allocation(R, alpha, C, n):
    revenue_values = compute_revenue_values(R, alpha, n, C)
    dp = np.zeros((n + 1, C + 1))
    allocation = np.zeros((n + 1, C + 1), dtype=int)
    data = []

    for i in range(1, n + 1):
        dt = TripleDict([f"Z={i}", f"phi_{i}(z_{i})", f"X_{i}"])
        for c in range(C + 1):
            dp[i, c] = dp[i - 1, c]
            if c > 0:
                values = [(dp[i - 1, c - x] + revenue_values[i - 1][x], x) for x in range(1, c + 1)]
                dp[i, c], allocation[i, c] = max(values, key=lambda pair: pair[0])
            dt.add(c, dp[i, c], allocation[i, c])
        data.append(dt)

    optimal_allocation = np.zeros(n, dtype=int)
    remaining_resources = C
    for i in range(n, 0, -1):
        optimal_allocation[i - 1] = allocation[i, remaining_resources]
        remaining_resources -= optimal_allocation[i - 1]

    optimal_revenue = dp[n, C]
    return optimal_allocation, optimal_revenue, dp, allocation, data

def print_results(optimal_allocation, R, alpha, C, max_revenue):
    print("\nOptimal Resource Allocation:")
    for i, x in enumerate(optimal_allocation):
        revenue = R[i] * (1 - (1 - np.exp(-alpha[i] / x)) ** x) if x > 0 else 0
        print(f"Product {i + 1}: {x} resources -> Revenue: {revenue:.2f}")
    print(f"Total resources used: {sum(optimal_allocation)} out of {C}")
    print(f"Maximum revenue: {max_revenue:.2f}")

# Example Parameters
R = np.array([5, 4, 10, 8, 3, 7, 2, 8, 3])
alpha = np.array([3, 4, 5.5, 3, 4.5, 2.5, 4, 5, 4.5])
C = 36
n = 9

# Run the model
optimal_allocation, max_revenue, dp, allocation, data = solve_resource_allocation(R, alpha, C, n)
print_results(optimal_allocation, R, alpha, C, max_revenue)

# Plotting
plot_allocation(allocation, n, C)
plot_optimal_allocation(optimal_allocation)
