import os

import numpy as np
import json

import pandas as pd


def feature_names(info, labels=None):
    features = [
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
    if labels is not None:
        with open("labels/"+labels+".txt", "r") as f:
            labels = list(f)
        labels = [l[:-1] for l in labels]
        features = labels + features

    if info["context features"]:
        features = ["goals found", "marked states found", "pot winning loops found", "frontier / explored"]+features
    
    if info["ra feature"]:
        features = ["ra type", "1 / ra distance", "in open"]+features
    return features


def results_path(problem, n, k, file):
    return "experiments/results/"+filename([problem, n, k])+"/"+file


def fsp_path(problem, n, k):
    return "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp"


def agent_path(problem, n, k, dir, idx):
    return "experiments/results/" + filename([problem, n, k]) + "/" + dir + "/" + str(idx) + ".onnx"


def filename(parameters):
    return "_".join(list(map(str, parameters)))


def best_agent_idx(problem, train_n, train_k, file):
    path = "experiments/results/" + filename([problem, train_n, train_k]) + "/" + file + "/"
    df = pd.read_csv(path + problem+"_3_3.csv")
    return df.loc[df["expanded transitions"] == df["expanded transitions"].min()]["idx"].iloc[-1]

    #ranks = [[] for _ in range(101)]
    #for f in os.listdir(path):
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

    #print(ranks)
    #print(list(map(np.prod, ranks)))
    #print(ranks[np.argmin(list(map(np.prod, ranks)))])
    #return np.argmin(list(map(np.prod, ranks)))


def last_agent_idx(df):
    return df["idx"].max()


def get_agent_info(path):
    with open(path[:-5]+".json", "r") as f:
        info = json.load(f)
    return info

def uses_context(path):
    info = get_agent_info(path)
    return "context features" in info.keys() and info["context features"]


def uses_ra(path):
    info = get_agent_info(path)
    return "ra feature" in info.keys() and info["ra feature"]


def uses_labels(path):
    info = get_agent_info(path)
    return "labels" in info.keys() and info["labels"]


def read_results(lines):
    results = {}
    results["expanded states"] = int(lines[0].split(" ")[1])
    results["used states"] = int(lines[1].split(" ")[1])
    results["expanded transitions"] = int(lines[2].split(" ")[1])
    results["used transitions"] = int(lines[3].split(" ")[1])
    results["synthesis time(ms)"] = int(lines[4].split(" ")[3])
    findNewLine = lines[5].split(" ")
    results["findNewGoals"] = int(findNewLine[1][:-1])
    results["findNewErrors"] = int(findNewLine[3])
    propagateLine = lines[6].split(" ")
    results["propagateGoals"] = int(propagateLine[1][:-1])
    results["propagateErrors"] = int(propagateLine[3])
    results["memory(mb)"] = float(lines[7].split(" ")[1])
    results["heuristic time(ms)"] = float(lines[8].split(" ")[1]) if "heuristic" in lines[8] else np.nan
    return results
