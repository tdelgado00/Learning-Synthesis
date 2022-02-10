from util import *
from prettytable import PrettyTable
import numpy as np

monolithic_results = get_problem_data("mono")["expandedTransitions"]

problems = ["TA", "TL", "AT", "BW", "CM", "DP"]

print("Cada fila es un valor de n")
for problem in problems:
    t = PrettyTable()
    t.field_names = [problem] + [str(i) for i in range(1, 6)]
    for n in range(1, 6):
        row = [str(n)]
        for k in range(1, 6):
            try:
                # row.append(monolithic_results[(problem, n, k)])
                row.append(np.round(np.log(monolithic_results[(problem, n, k)]), 2))
            except:
                row.append("-")
        t.add_row(row)
    print(t)
