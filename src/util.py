import numpy as np
import json


def feature_names(ra_feature):
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
    if ra_feature:
        features = ["ra type best", "1 / ra distance best", "ra type last", "1 / ra distance last", "in open"]+features
    return features


def results_path(problem, n, k, file):
    return "experiments/results/"+filename([problem, n, k])+"/"+file


def fsp_path(problem, n, k):
    return "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp"


def agent_path(problem, n, k, dir, idx):
    return "experiments/results/" + filename([problem, n, k]) + "/" + dir + "/" + str(idx) + ".onnx"


def filename(parameters):
    return "_".join(list(map(str, parameters)))


def best_agent_idx(df):
    return df.loc[df["expanded transitions"] == df["expanded transitions"].min()]["idx"].iloc[0]


def last_agent_idx(df):
    return df["idx"].max()


def get_agent_info(path):
    with open(path[:-5]+".json", "r") as f:
        info = json.load(f)
    return info


def uses_ra(path):
    info = get_agent_info(path)
    return "ra feature" in info.keys() and info["ra feature"]


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
