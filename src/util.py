import numpy as np
import pandas as pd

feature_names = [
    "controllable",
    "depth",
    "state unexplorability",
    "state marked",
    "child marked",
    "child goal",
    "child error",
    "child none",
    "child deadlock",
    "uncontrollability child",
    "unexplorability child",
]

def filename(parameters):
    return "_".join(list(map(str, parameters)))


def get_problem_data(algorithm, heuristic=None):
    df = pd.DataFrame()
    for problem in ["TA", "TL", "AT", "BW", "CM", "DP"]:
        curr_df = pd.read_csv("experiments/results/ResultsPaper/" + problem + "_mejorado.csv")
        curr_df = curr_df.loc[(curr_df["controllerType"] == algorithm) &
                              (heuristic is None or curr_df["heuristic"] == heuristic)]
        df = df.append(curr_df)
    df["testcase"] = df["testcase"].apply(lambda t: t.split("-"))
    df["testcase"] = df["testcase"].apply(lambda t: (t[0], int(t[1]), int(t[2])))
    df = df.set_index("testcase").to_dict()
    return df


def read_results(proc):
    lines = proc.stdout.split("\n")
    if "seed" in lines[0]:
        lines = lines[1:]
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
    results["heuristic time(ms)"] = float(lines[8].split(" ")[1])
    return results
