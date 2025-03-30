import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import importlib
import simplex_module as simplex

# Reload module to reflect changes
importlib.reload(simplex)

def solve_simplex():
    try:
        c = np.array([float(x) for x in entry_c.get().split()])
        a = np.array([[float(x) for x in row.split()] for row in entry_a.get("1.0", "end").strip().split("\n")])
        b = np.array([float(x) for x in entry_b.get().split()])
        equalities = entry_equalities.get().split()
        max_problem = max_var.get()

        table = simplex.simplex_method(c, a, b, equalities, max_problem, integer=True)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Simplex Table:\n{table}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("Simplex Solver")

# Objective Function
ttk.Label(root, text="Objective Function Coefficients (c):").grid(row=0, column=0)
entry_c = ttk.Entry(root, width=50)
entry_c.grid(row=0, column=1)

# Constraints
ttk.Label(root, text="Constraint Coefficients Matrix (A):").grid(row=1, column=0)
entry_a = tk.Text(root, width=50, height=5)
entry_a.grid(row=1, column=1)

# RHS Values
ttk.Label(root, text="Right-Hand Side Values (b):").grid(row=2, column=0)
entry_b = ttk.Entry(root, width=50)
entry_b.grid(row=2, column=1)

# Equalities
ttk.Label(root, text="Inequalities (<=, >=, =):").grid(row=3, column=0)
entry_equalities = ttk.Entry(root, width=50)
entry_equalities.grid(row=3, column=1)

# Max or Min Problem
max_var = tk.BooleanVar()
ttMax = ttk.Checkbutton(root, text="Maximization Problem", variable=max_var)
ttMax.grid(row=4, column=1)

# Solve Button
solve_button = ttk.Button(root, text="Solve", command=solve_simplex)
solve_button.grid(row=5, column=1)

# Result Box
ttLabel = ttk.Label(root, text="Result:").grid(row=6, column=0)
result_text = tk.Text(root, width=100, height=10)
result_text.grid(row=6, column=1)

root.mainloop()