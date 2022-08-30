import os

import numpy as np
import json

import pandas as pd


def feature_names(info, problem=None):
    base_features = [
        "action controllable",
        "1 / depth",
        "state portion explored",
        "state portion controllable",
        "state marked",
        "child marked",
        "child goal",
        "child error",
        "child none",
        "child deadlock",
        "child portion controllable",
        "child portion explored",
    ]
    context_features = ["goals found", "marked states found", "pot winning loops found", "frontier / explored"]
    ra_features = ["ra type", "1 / ra distance", "in open"]
    je_features = ["last state expanded from", "last state expanded to"]
    nk_features = ["n", "k"]
    prop_features = ["in loop", "forced in FNG", "child in loop", "child forced in FNG", "in PG ancestors", "forced in PG", "in forced to clausure"]
    visits_features = ["expl percentage", "state visits", "child visits"]

    check = lambda p : p in info.keys() and info[p]
    ra = check("ra feature")
    labels = check("labels")
    state_labels = check("state labels")
    context = check("context features")
    je = check("je feature")
    nk = check("nk feature")
    p = check("prop feature")
    v = check("visits feature")

    if problem is not None and labels:
        with open("labels/" + problem + ".txt", "r") as f:
            labels = list(f)
        labels_features = [l[:-1] for l in labels]

    features = (ra_features if ra else []) + \
               (context_features if context else []) + \
               (["state " + l for l in labels_features] if labels else []) + \
               (labels_features if state_labels else []) + \
               base_features + \
               (je_features if je else []) + \
               (nk_features if nk else []) + \
               (prop_features if p else []) + \
               (visits_features if v else [])

    return features


def results_path(problem, n=2, k=2, file=""):
    return "experiments/results/" + filename([problem, n, k]) + "/" + file


def fsp_path(problem, n, k):
    return "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp"


def agent_path(dir, idx):
    return "experiments/results/" + dir + "/" + str(idx) + ".onnx"


def filename(parameters):
    return "_".join(list(map(str, parameters)))


def best_agent_idx(problem, train_n, train_k, file):
    path = "experiments/results/" + filename([problem, train_n, train_k]) + "/" + file + "/"
    df = pd.read_csv(path + problem + "_3_3.csv")
    return df.loc[df["expanded transitions"] == df["expanded transitions"].min()]["idx"].iloc[-1]


def last_agent_idx(df):
    return df["idx"].max()


def get_agent_info(path):
    with open(path[:-5] + ".json", "r") as f:
        info = json.load(f)
    return info


def uses_feature(path, feature):
    info = get_agent_info(path)
    return feature in info.keys() and info[feature]

def indexOf(s, lines):
    return list(map(lambda l: s in l, lines)).index(True)


def read_results(lines):
    i = indexOf("ExpandedStates", lines)
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
    return results


def best_agent_2_2(problem, file):
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")
    df = df.loc[(df["n"] == 2) & (df["k"] == 2)]
    m = df["expanded transitions"].min(skipna=True)
    df = df.loc[df["expanded transitions"] == m]
    idxs = list(df["idx"])
    print(idxs)
    return max(idxs)

def best_generalization_agent(problem, file):
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")
    max_idx = df["idx"].max()
    solved = [0 for i in range(max_idx + 1)]
    expanded = [0 for i in range(max_idx + 1)]
    for x, cant in dict(df["idx"].value_counts()).items():
        solved[x] = cant
    for x, cant in dict(df.groupby("idx")["expanded transitions"].sum()).items():
        expanded[x] = cant
    perf = [(solved[i], expanded[i], i) for i in range(max_idx + 1)]
    return max(perf, key=lambda t: (t[0], -t[1], t[2]))[2]


def train_instances(problem, max_size=10000):
    r = monolithic_results["expanded transitions", problem]
    instances = []
    for n in range(2, 16):
        for k in range(2, 16):
            if not np.isnan(r[k][n]) and r[k][n] <= max_size:
                instances.append((problem, n, k))
    return instances

def all_solved_instances(dfs):
    instances = []
    for n in range(1, 16):
        for k in range(1, 16):
            good = True
            for df, cant in dfs:
                if ((df["n"] == n) & (df["k"] == k)).sum() < cant:
                    good = False
                    break
            if good:
                instances.append((n, k))
    return instances