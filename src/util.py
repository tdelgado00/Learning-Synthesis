import numpy as np
import json
import sys
import pandas as pd
import pickle


def results_path(problem, n=2, k=2, file=""):
    return "experiments/results/" + filename([problem, n, k]) + "/" + file


def fsp_path(problem, n, k):
    return "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp"


def agent_path(problem, file, idx, n=2, k=2):
    return results_path(problem, n, k, file) + "/" + str(idx) + ".onnx"


def filename(parameters):
    return "_".join(list(map(str, parameters)))


def last_agent(df):
    return df["idx"].max()
def joinAsStrings(listOfArgs):
    res = ""
    for arg in listOfArgs: res+=("_"+str(arg))
    return res

def best_agent_2_2(problem, file):
    return best_agent_n_k(problem, file, 2, 2)


def best_agent_3_3(problem, file):
    return best_agent_n_k(problem, file, 3, 3)


def best_agent_n_k(problem, file, n=2, k=2):
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")
    df = df.loc[(df["n"] == n) & (df["k"] == k)]
    m = df["expanded transitions"].min(skipna=True)
    df = df.loc[df["expanded transitions"] == m]
    idxs = list(df["idx"])
    return max(idxs)


def best_generalization_agent(problem, file):
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")
    max_idx = df["idx"].max()
    solved = [0 for _ in range(max_idx + 1)]
    expanded = [0 for _ in range(max_idx + 1)]
    for x, cant in dict(df["idx"].value_counts()).items():
        solved[x] = cant
    for x, cant in dict(df.groupby("idx")["expanded transitions"].sum()).items():
        expanded[x] = cant
    perf = [(solved[i], expanded[i], i) for i in range(max_idx + 1)]
    return max(perf, key=lambda t: (t[0], -t[1], t[2]))[2]


def get_agent_info(path):
    with open(path[:-5] + ".json", "r") as f:
        info = json.load(f)
    return info


def uses_feature(path, feature):
    info = get_agent_info(path)
    return feature in info.keys() and info[feature]


def read_results(lines):
    def indexOf(s):
        return list(map(lambda l: s in l, lines)).index(True)

    i = indexOf("ExpandedStates")
    results = {}
    results["expanded states"] = int(lines[i].split(" ")[1])
    results["used states"] = int(lines[i + 1].split(" ")[1])
    results["expanded transitions"] = int(lines[i + 2].split(" ")[1])
    results["used transitions"] = int(lines[i + 3].split(" ")[1])
    results["synthesis time(ms)"] = int(lines[i + 4].split(" ")[3])
    findNewLine = lines[i + 5].split(" ")
    results["findNewGoals"] = int(findNewLine[1][:-1])
    results["findNewErrors"] = int(findNewLine[3])
    propagateLine = lines[i + 6].split(" ")
    results["propagateGoals"] = int(propagateLine[1][:-1])
    results["propagateErrors"] = int(propagateLine[3])
    results["memory(mb)"] = float(lines[i + 7].split(" ")[1])
    results["heuristic time(ms)"] = float(lines[i + 8].split(" ")[1]) if "heuristic" in lines[i + 8] else np.nan
    results["expansion_budget_exceeded"] = (lines[i + 9].split(" ")[1])
    return results


def best_generalization_agent_ebudget(problem, file,up_to, timeout,total,ebudget):
    df = pd.read_csv("experiments/results/" + filename(
        [problem, 2, 2]) + "/" + file + "/generalization_all"+ joinAsStrings([up_to, timeout,total,ebudget])+".csv")
    df = notExceeded(df)
    if df.shape[0]==0:
        return None
    max_idx = df["idx"].max()
    solved = [0 for i in range(max_idx + 1)]
    expanded = [0 for i in range(max_idx + 1)]
    for x, cant in dict(df["idx"].value_counts()).items():
        solved[x] = cant
    for x, cant in dict(df.groupby("idx")["expanded transitions"].sum()).items():
        expanded[x] = cant
    perf = [(solved[i], expanded[i], i) for i in range(max_idx + 1)]
    return max(perf, key=lambda t: (t[0], -t[1], t[2]))[2]


def solved_by_agent(path):
    df = pd.read_csv(path)
    df = notExceeded(df)
    return df["heuristic"].value_counts(sort=False)


def notExceeded(df):
    return df[df["expansion_budget_exceeded"] == False]


def fill_df(df, m):
    added = []
    for n in range(1, m + 1):
        for k in range(1, m + 1):
            if len(df.loc[(df["n"] == n) & (df["k"] == k)]) == 0:
                added.append({"n": n, "k": k, "expanded transitions": float("inf"), "synthesis time(ms)": float("inf")})
    df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    return df


def parse_java_debug(debug):
    """ This function parses the output of the synthesis procedure when -d flag is used """
    debug = [l for l in debug if l != "--------------------------"]
    steps = []
    i = 0
    while i < len(debug):
        first = debug[i].split(" ")
        n, nfeatures = int(first[0]), int(first[1])
        actions = []
        features = []
        for j in range(n):
            line = debug[i + 1 + j].split(" ")
            actions.append(line[0])
            features.append([float(x) for x in line[1:nfeatures + 1]])
        values = [float(x) for x in debug[i + 1 + n:i + 1 + 2 * n]]
        selected = float(debug[i + 1 + 2 * n])
        steps.append(
            {"frontier size": n, "nfeatures": nfeatures, "actions": actions, "features": features, "values": values,
             "selected": selected})
        i += n * 2 + 2
    return steps


def train_instances(problem, max_size=10000):
    """ Function used to observe the set of instances smaller than a certain size """
    r = read_monolithic()["expanded transitions", problem]
    instances = []
    for n in range(2, 16):
        for k in range(2, 16):
            if not np.isnan(r[k][n]) and r[k][n] <= max_size:
                instances.append((problem, n, k))
    return instances


def read_monolithic():
    monolithic_results = {}
    for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
        df = pd.read_csv("experiments/results/ResultsPaper/" + problem + ".csv")
        df = df.loc[df["controllerType"] == "mono"]
        df["n"] = df["testcase"].apply(lambda t: int(t.split("-")[1]))
        df["k"] = df["testcase"].apply(lambda t: int(t.split("-")[2]))
        df["expanded transitions"] = df["expandedTransitions"]
        df = df.dropna(subset=["expanded transitions"])
        df = fill_df(df, 15)
        monolithic_results["expanded transitions", problem] = df.pivot("n", "k", "expanded transitions")
        monolithic_results["synthesis time(ms)", problem] = df.pivot("n", "k", "synthesisTimeMs")
    return monolithic_results

def budget_and_time(series):
    return series["expansion_budget_exceeded"] == 'false' and not np.isnan(series["synthesis time(ms)"])

