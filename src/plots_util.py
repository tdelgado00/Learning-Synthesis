import numpy as np
import json

import pandas as pd
import pickle
from util import *


def feature_names(info, problem=None):
    check = lambda p: p in info.keys() and info[p]
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
    prop_features = ["in loop", "forced in FNG", "child in loop", "child forced in FNG", "in PG ancestors",
                     "forced in PG", "in forced to clausure"]
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


def read_test(data, problems, files, name="/generalization_all.csv"):
    print("Reading test files")

    data["all"] = {}

    old_files = ["base_features_2h", "ra_feature2opt_2h", "5mill_RA", "5mill_L"]

    for p in problems:
        data["all"][p] = {}
        for file, group in files:
            if file in old_files:
                path = results_path(p) + file + "/" + filename([p, 2, 2]) + ".csv"
            else:
                path = results_path(p) + file + name
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
            # if not ("focused" in file or "epsdec" in file or "5kk" in file):
            idxs = list(df["idx"].unique())
            df["idx"] = df["idx"].apply(lambda i: idxs.index(i))
            df["expanded transitions / total"] = df["expanded transitions"] / df["total transitions"]
            df["min transitions"] = df.groupby("instance")["expanded transitions"].cummin()
            data["all"][p][file] = df


def read_ra_and_random(used_problems, data):
    data.update({name: {} for name in
                 ["ra 5s", "random 5s", "ra 10m", "random 10m", "ra 30m", "ra 15000t", "ra 5000t", "random 15000t"]})
    for p in used_problems:
        data["ra 5s"][p] = pd.read_csv(results_path(p) + "/RA_5s_15.csv")
        data["random 5s"][p] = pd.read_csv(results_path(p) + "/random_5s.csv")
        data["ra 15000t"][p] = pd.read_csv(results_path(p) + "/all_ra_15000t.csv")
        data["ra 5000t"][p] = pd.DataFrame(
            data["ra 15000t"][p].loc[data["ra 15000t"][p]["expanded transitions"] < 5001])
        data["ra 10m"][p] = pd.read_csv(results_path(p) + "/all_ra_afterfix_15.csv")
        data["ra 30m"][p] = pd.read_csv(results_path(p) + "/all_ra_30m_15.csv")
        data["random 10m"][p] = [pd.read_csv(results_path(p) + "random/all random " + str(i) + ".csv") for i in
                                 range(5)]
        data["random 15000t"][p] = [pd.read_csv(results_path(p) + "all_random" + str(i) + "_1500015000.csv") for i in
                                    range(1, 6, 1)]

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


def read_agents_evaluation(problems, files, name="all"):
    agents_10m = {}
    for p in problems:
        agents_10m[p] = {}
        for file, g in files:
            agents_10m[p][file] = pd.read_csv(results_path(p, file=file) + "/" + "all_"+ p +"_15_15000_TO:10h.csv")
    return agents_10m
