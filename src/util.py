import os

import numpy as np
import json

import pandas as pd


def onlyifsolvedlast(res):
    for n in range(1, 16):
        for k in range(1, 16):
            if (n > 1 and res[n - 1][k] == float("inf")) or (k > 1 and res[n][k - 1] == float("inf")):
                res[n][k] = float("inf")
    return res


def fill_df(df, m):
    added = []
    for n in range(1, m + 1):
        for k in range(1, m + 1):
            if len(df.loc[(df["n"] == n) & (df["k"] == k)]) == 0:
                added.append({"n": n, "k": k, "expanded transitions": float("inf"), "synthesis time(ms)": float("inf")})
    df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    return df


def df_agent(problem, agent_file, metric="expanded transitions"):
    df_agent = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + agent_file)
    # df_agent = pd.read_csv("experiments/results 25 mar/"+filename([problems[i], 2, 2])+"/all_e_15.csv")
    df_agent = fill_df(df_agent, 15)
    agent_t = df_agent.pivot("n", "k", metric)
    agent_t = agent_t.fillna(float("inf"))
    r = onlyifsolvedlast(agent_t)
    return r


def df_comp(problem, comp_df, metric="expanded transitions"):
    comp_t = comp_df[metric, problem]
    comp_t = comp_t.fillna(float("inf"))
    return onlyifsolvedlast(comp_t)


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


def results_path(problem, n, k, file):
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

    # ranks = [[] for _ in range(101)]
    # for f in os.listdir(path):
    #    if f.endswith(".csv") and len(f.split("_")) == 3:
    #        df = pd.read_csv(path + f)
    #        n, k = tuple([int(x) for x in f[:-4].split("_")[1:]])
    #        if n != train_n or k != train_k:
    #            models = []
    #            for idx, row in df.iterrows():
    #                if row["idx"] == 70:
    #                    print("Best dp", row["expanded transitions"], n, k)
    #                models.append((row["expanded transitions"], row["idx"]))
    #            models = sorted(models)
    #            for i in range(len(models)):
    #                ranks[models[i][1]].append(i+1)

    # print(ranks)
    # print(list(map(np.prod, ranks)))
    # print(ranks[np.argmin(list(map(np.prod, ranks)))])
    # return np.argmin(list(map(np.prod, ranks)))


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


monolithic_results = {}
for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    df = pd.read_csv("experiments/results/ResultsPaper/" + problem + ".csv")
    df = df.loc[df["controllerType"] == "mono"]
    df["n"] = df["testcase"].apply(lambda t: int(t.split("-")[1]))
    df["k"] = df["testcase"].apply(lambda t: int(t.split("-")[2]))
    df = fill_df(df, 15)
    monolithic_results["expanded transitions", problem] = df.pivot("n", "k", "expandedTransitions")
    monolithic_results["synthesis time(ms)", problem] = df.pivot("n", "k", "synthesisTimeMs")

ra_results = {}
for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/all_ra_afterfix_15.csv")
    df = fill_df(df, 15)
    ra_results["expanded transitions", problem] = df.pivot("n", "k", "expanded transitions")
    ra_results["synthesis time(ms)", problem] = df.pivot("n", "k", "synthesis time(ms)")

ra_sola_results = {}
for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/all_ra_sola_15.csv")
    df = fill_df(df, 15)
    ra_sola_results["expanded transitions", problem] = df.pivot("n", "k", "expanded transitions")
    ra_sola_results["synthesis time(ms)", problem] = df.pivot("n", "k", "synthesis time(ms)")

random_results_small = {}
for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    for n, k in [(2, 2), (3, 3)]:
        df = pd.read_csv("experiments/results/" + filename([problem, n, k]) + "/random.csv")
        random_results_small[(problem, n, k)] = list(df["expanded transitions"])

random_results = {}
for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/all_random.csv")
    df = fill_df(df, 15)
    random_results["expanded transitions", problem] = df.pivot("n", "k", "expanded transitions")
    random_results["synthesis time(ms)", problem] = df.pivot("n", "k", "synthesis time(ms)")
