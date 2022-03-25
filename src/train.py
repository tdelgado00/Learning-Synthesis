import json
import os
import pickle

import numpy as np
import onnx
import pandas as pd

from agent import Agent
from environment import DCSSolverEnv
from modelEvaluation import eval_agent_q, eval_agents_coefs
from test import test_agent, test_ra
from util import filename


def train_agent(problem, n, k, minutes, dir, eta=1e-6, epsilon=0.1, nnsize=20, copy_freq=200000, ra_feature=False):
    env = DCSSolverEnv(problem, n, k, ra_feature)
    print("Number of features:", env.nfeatures)
    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir if dir is not None else None

    agent = Agent(eta=eta, nnsize=nnsize, epsilon=epsilon, dir=dir)
    agent.train(env, minutes * 60, copy_freq=copy_freq, agent_info={"ra feature": ra_feature})

    return agent


def test_agents(problem, n, k, file, problems, freq=1):
    with open("experiments/results/" + filename([problem, n, k]) + "/" + file + "_states.pkl", "rb") as f:
        random_states = pickle.load(f)

    df = []
    dir = "experiments/results/" + filename([problem, n, k]) + "/" + file
    for i in range(len(os.listdir(dir)) // 2):
        if i % freq != 0:
            continue
        agent = onnx.load(dir + "/" + str(i) + ".onnx")

        with open(dir + "/" + str(i) + ".json", "r") as f:
            info = json.load(f)

        avg_q = eval_agent_q(agent, random_states)

        coefs = eval_agents_coefs(agent, random_states, info)

        for problem2, n2, k2 in problems:
            print("Testing", i, "with", problem2, n2, k2)
            result, debug = test_agent(problem2, n2, k2, timeout="10m", dir=file, idx=i)
            if result == "timeout":
                result = {"problem": problem2, "n": n2, "k": k2}
            result.update(info)
            result["avg q"] = avg_q
            result["idx"] = i
            result.update(coefs)
            df.append(result)

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, n, k]) + "/" + file + ".csv")


def exp_hyp(problem, n, k, minutes):
    for eta in [1e-3, 1e-4, 1e-5, 1e-6]:
        for eps in [0.05, 0.1, 0.2]:
            for it in range(3):
                train_agent(problem, n, k, minutes, "eta_" + str(eta), eta=eta, epsilon=eps)


def exp_nnsize(problem, n, k, minutes):
    for nnsize in [5, 10, 20, 50, 100]:
        train_agent(problem, n, k, minutes, "nnsize_" + str(nnsize), nnsize=nnsize)


def exp_test_all_ra(problem, up_to, old=False, timeout="10m", name="all_ra"):
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if n == 0 or solved[n - 1][k] or k == 0 or solved[n][k - 1]:
                print("Testing ra with", problem, n, k, "- Old:", old)
                df.append(test_ra(problem, n + 1, k + 1, timeout=timeout, old=old)[0])
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True

    df = pd.DataFrame(df)
    file = filename([name, up_to]) + (".csv" if not old else "_old.csv")
    df.to_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file)


agent_idx = {
    "AT": 95,
    "TA": 105,
    "TL": 105,
    "BW": 95,
    "DP": 130
}

if __name__ == "__main__":

    for problem in ["AT", "BW", "TL", "DP", "TA"]:
        train_agent(problem, 2, 2, 20, "base_features", copy_freq=10000, ra_feature=False)
        test_agents(problem, 2, 2, "base_features", [(problem, 2, 2), (problem, 3, 3)])
        train_agent(problem, 2, 2, 0.2, "ra_feature", copy_freq=2000, ra_feature=True)
        test_agents(problem, 2, 2, "ra_feature", [(problem, 2, 2), (problem, 3, 3)])