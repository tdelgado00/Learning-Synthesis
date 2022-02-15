import subprocess

import numpy as np
import pandas as pd
from util import read_results


def test(problem, n, k, heuristic, timeout="30m", old=False):
    if old:
        print("Running old RA with", problem, n, k)
        jar = "mtsaOld.jar"
    else:
        print("Running RA with", problem, n, k)
        jar = "mtsa.jar"
    path = "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp"
    proc = subprocess.run(["timeout", timeout, "java", "-Xmx8g", "-classpath", jar,
                           "ltsa.ui.LTSABatch", "-i", path, "-c", "DirectedController"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        results = read_results(proc)
    results["algorithm"] = "old" if old else "new"
    results["heuristic"] = heuristic
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    print("Done.", results["synthesis time(ms)"])
    return results


def test_mono(problem, n, k):
    proc = subprocess.Popen(
        ["timeout", "30m", "java", "-cp", "mtsaOld.jar", "ltsa.ui.LTSABatch", "-i",
         "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp", "-c", "MonolithicController"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    results = read_results(proc)
    results["algorithm"] = "mono"
    results["heuristic"] = None
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results


def test_all():
    df = []
    for problem in ["TA", "TL", "AT", "BW", "CM", "DP"]:
        for n in range(1, 6):
            for k in range(1, 6):
                df.append(test(problem, n, k, "r"))
                df.append(test(problem, n, k, "m"))
                df.append(test(problem, n, k, "b"))
                df.append(test(problem, n, k, "t"))
                df.append(test_mono(problem, n, k))
                df.append(test_old(problem, n, k, "r"))
                df.append(test_old(problem, n, k, "m"))
                df.append(test_old(problem, n, k, "b"))
    df = pd.DataFrame(df)
    df.to_csv("../experiments/results/current_results.csv")


def find_inefficiency():
    df = pd.read_csv("results.csv")
    df = df.loc[df.n == 1]
    df = df.loc[df.k == 1]
    df = df.loc[df.heuristic == "r"]
    problems = ["TA", "TL", "AT", "BW", "CM", "DP"]
    for problem in problems:
        dfp = df.loc[df.problem == problem]
        new = dfp.loc[dfp.algorithm == "new"]["expanded transitions"].max()
        old = dfp.loc[dfp.algorithm == "old"]["expanded transitions"].max()
        print(problem, new, old)


if __name__ == '__main__':
    print(test("AT", 3, 1, "r", old=False))
    print(test("AT", 3, 1, "r", old=True))
