from util import *
from prettytable import PrettyTable

problems = ["TA", "TL", "AT", "BW", "CM", "DP"]
print("Cada fila es un valor de n")
for problem in problems:

    df = []
    t = PrettyTable()
    t.field_names = [problem] + [str(i) for i in range(1, 16)]
    for n in range(1, 16):
        row = [str(n)]
        for k in range(1, 16):
            try:
                df.append({"n": n, "k": k, "transitions": monolithic_results["expanded transitions", problem][k][n]})
                row.append(monolithic_results["expanded transitions", problem][k][n])
                #row.append(np.round(np.log(monolithic_results[(problem, n, k)]), 2))
            except:
                row.append("-")
        t.add_row(row)
    print(t)
    #print(pd.DataFrame(df).pivot("n", "k", "transitions").to_csv())

