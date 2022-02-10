import subprocess

import pandas as pd
from util import read_results


def test(problem, n, k, heuristic):
    proc = subprocess.Popen(
        ["timeout", "30m", "java", "-cp", "mtsa.jar", "ltsa.ui.LTSABatch", "-i",
         "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp", "-c", "DirectedController",
         "-" + heuristic],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    results = read_results(proc)
    results["algorithm"] = "new"
    results["heuristic"] = heuristic
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results


def test_old(problem, n, k, heuristic):
    proc = subprocess.Popen(
        ["timeout", "30m", "java", "-cp", "mtsaOld.jar", "ltsa.ui.LTSABatch", "-i",
         "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp", "-c", "DirectedController",
         "-" + heuristic],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    results = read_results(proc)
    results["algorithm"] = "old"
    results["heuristic"] = heuristic
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
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
    print(test("AT", 2, 2, "r"))
