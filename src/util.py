import numpy as np
import json

import pandas as pd
import pickle


def feature_names(info, problem=None):
    check = lambda p : p in info.keys() and info[p]
    if check("only boolean"):
        base_features = [
            "action controllable",
            "state trans explored",
            "state unexp uncontrollable",
            "state uncontrollable",
            "state marked",
            "child marked",
            "child goal",
            "child error",
            "child none",
            "child deadlock",
            "child unexp uncontrollable",
            "child uncontrollable",
            "child trans explored",
        ]
        context_features = ["goal found", "marked found", "loop closed"]
    else:
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


def agent_path(problem, file, idx, n=2, k=2):
    return results_path(problem, n, k, file) + "/" + str(idx) + ".onnx"


def filename(parameters):
    return "_".join(list(map(str, parameters)))


def last_agent(df):
    return df["idx"].max()


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
    return results


def train_instances(problem, max_size=10000):
    r = read_monolithic()["expanded transitions", problem]
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


def read_test(data, problems, files):
    print("Reading test files")

    data["all"] = {}

    old_files = ["base_features_2h", "ra_feature2opt_2h", "5mill_RA", "5mill_L"]

    for p in problems:
        data["all"][p] = {}
        for file, group in files:
            if file in old_files:
                path = results_path(p) + file + "/" + filename([p, 2, 2]) + ".csv"
            else:
                path = results_path(p) + file + "/generalization_all.csv"
            try:
                df = pd.read_csv(path)
            except:
                print("File not found", path)
                continue
            df = df.dropna(subset=["expanded transitions"])
            df["instance"] = df.apply((lambda r: (r["problem"], r["n"], r["k"])), axis=1)
            df["total transitions"] = df.apply(
                lambda r: data["mono"]["expanded transitions", r["problem"]][r["k"]][r["n"]], axis=1)
            df["group"] = group
            df["file"] = file
            idxs = list(df["idx"].unique())
            df["idx"] = df["idx"].apply(lambda i: idxs.index(i))
            df["expanded transitions / total"] = df["expanded transitions"] / df["total transitions"]
            df["min transitions"] = df.groupby("instance")["expanded transitions"].cummin()
            data["all"][p][file] = df


def read_ra_and_random(used_problems, data):
    data.update({name: {} for name in ["ra 5s", "random 5s", "ra 10m", "random 10m", "ra 30m"]})
    for p in used_problems:
        data["ra 5s"][p] = pd.read_csv(results_path(p) + "/RA_5s_15.csv")
        data["random 5s"][p] = pd.read_csv(results_path(p) + "/random_5s.csv")

        data["ra 10m"][p] = pd.read_csv(results_path(p) + "/all_ra_afterfix_15.csv")
        data["ra 30m"][p] = pd.read_csv(results_path(p) + "/all_ra_30m_15.csv")
        data["random 10m"][p] = [pd.read_csv(results_path(p) + "random/all random "+str(i)+".csv") for i in range(5)]

        for l in [data[name][p] for name in ["ra 5s", "random 5s", "ra 10m", "random 10m", "ra 30m"]]:
            for df in [l] if type(l) != list else l:
                df["instance"] = df.apply(lambda r: (r["problem"], r["n"], r["k"]), axis=1)
                df["total transitions"] = df.apply(
                    lambda r: data["mono"]["expanded transitions", r["problem"]][r["k"]][r["n"]], axis=1)
                df.dropna(subset=["expanded transitions"])


def read_training_for_file(problem, file, multiple):
    if file == "5mill_JE_NORA":  # we didn't save training data
        return None
    with open(results_path(problem, file=file) + "/training_data.pkl", "rb") as f:
        training_data, agent_params, env_params = pickle.load(f)
    df = pd.DataFrame(training_data)
    df["file"] = file
    df["problem"] = problem
    df["multiple"] = multiple
    if multiple:
        instances = train_instances(problem)
        df["n"] = df.apply(lambda r: instances[int(r["idx"]) % len(instances)][1], axis=1)
        df["k"] = df.apply(lambda r: instances[int(r["idx"]) % len(instances)][2], axis=1)
    else:
        df["n"] = 2
        df["k"] = 2
    return df


def read_training(data, used_problems, used_files, base=10000, window_size=10):
    print("Reading training")
    data.update({"train": {}, "bucket train": {}})
    for problem in used_problems:
        data["train"][problem] = {}
        for file, group in used_files:
            multiple = "RR" in group
            df = read_training_for_file(problem, file, multiple)
            if df is None:
                continue
            df["group"] = group
            data["train"][problem][file] = df


def read_random_small():
    random_results_small = {}
    for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
        for n, k in [(2, 2)]:
            df = pd.read_csv("experiments/results/" + filename([problem, n, k]) + "/random1.0")
            random_results_small[(problem, n, k)] = list(df["expanded transitions"])
    return random_results_small


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


def read_agents_10m(problems, files, name="all"):
    agents_10m = {}
    for p in problems:
        agents_10m[p] = {}
        for file, g in files:
            agents_10m[p][file] = pd.read_csv(results_path(p, file=file) + "/"+name+".csv")
    return agents_10m


def fill_df(df, m):
    added = []
    for n in range(1, m + 1):
        for k in range(1, m + 1):
            if len(df.loc[(df["n"] == n) & (df["k"] == k)]) == 0:
                added.append({"n": n, "k": k, "expanded transitions": float("inf"), "synthesis time(ms)": float("inf")})
    df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    return df