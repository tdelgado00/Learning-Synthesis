import json
import os
import pickle

import numpy as np
import onnx
import pandas as pd
from onnxruntime import InferenceSession
from sklearn.linear_model import LinearRegression

from agent import Agent, test_onnx
from environment import DCSSolverEnv
from test import test
from util import filename, get_problem_data, feature_names
max_actions = {}


def get_max_actions():
    r = get_problem_data("mono")["expandedTransitions"]
    for problem in ["AT", "TA", "DP", "TL", "BW", "CM"]:
        for n in range(1, 16):
            for k in range(1, 16):
                if (problem, n, k) in r.keys() and r[problem, n, k] == r[problem, n, k]:
                    max_actions[problem, n, k] = int(r[problem, n, k])
                else:
                    max_actions[problem, n, k] = 1000000


get_max_actions()


def eval_agents_coefs(agent, problem, n, k):
    with open("experiments/results/"+filename([problem, n, k])+"/random_states.pkl", "rb") as f:
        states = pickle.load(f)
    actions = np.array([a for s in states for a in s])

    sess = InferenceSession(agent.SerializeToString())

    values = sess.run(None, {'X': actions})[0]
    values = (values - np.mean(values)) / np.std(values)
    model = LinearRegression().fit(actions, values)
    coefs = {}
    for i in range(len(feature_names)):
        coefs[feature_names[i]] = model.coef_[0][i]
    return coefs


def eval_agent_q(agent, random_states):
    sess = InferenceSession(agent.SerializeToString())
    return np.mean([np.max(sess.run(None, {'X': s})) for s in random_states])


def train_agent(problem, n, k, minutes, dir, eta=1e-6, epsilon=0.1, nnsize=20, copy_freq=200000):
    env = DCSSolverEnv(problem, n, k, max_actions=max_actions[problem, n, k])

    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir if dir is not None else None

    agent = Agent(eta=eta, nnsize=nnsize, epsilon=epsilon, dir=dir)
    agent.train(env, minutes * 60, copy_freq=copy_freq)

    return agent


def test_agents(problem, n, k, file, problems, freq = 1):
    with open("experiments/results/"+filename([problem, n, k])+"/random_states.pkl", "rb") as f:
        random_states = pickle.load(f)

    df = []
    dir = "experiments/results/"+filename([problem, n, k])+"/"+file
    for i in range(len(os.listdir(dir))//2):
        if i % freq != 0:
            continue
        agent = onnx.load(dir+"/"+str(i)+".onnx")

        with open(dir+"/"+str(i)+".json", "r") as f:
            info = json.load(f)

        avg_q = eval_agent_q(agent, random_states)

        coefs = eval_agents_coefs(agent, problem, n, k)

        for problem2, n2, k2 in problems:
            env = DCSSolverEnv(problem2, n2, k2, max_actions=max_actions[problem2, n2, k2])
            print("Testing", i, "with", problem2, n2, k2)
            result = test_onnx(agent, env)
            if result == "timeout":
                result = {"problem": problem2, "n": n2, "k": k2}
            result.update(info)
            result["avg q"] = avg_q
            result["idx"] = i
            result.update(coefs)
            df.append(result)

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, n, k])+"/"+file+".csv")


def exp_hyp(problem, n, k, minutes):
    for eta in [1e-3, 1e-4, 1e-5, 1e-6]:
        for eps in [0.05, 0.1, 0.2]:
            for it in range(3):
                train_agent(problem, n, k, minutes, "eta_"+str(eta), eta=eta, epsilon=0.1)

def exp_nnsize(problem, n, k, minutes):
    for nnsize in [5, 10, 20, 50, 100]:
        train_agent(problem, n, k, minutes, "nnsize_"+str(nnsize), nnsize=nnsize)


def pick_agent(problem, n, k, file):
    df = pd.read_csv("experiments/results/"+filename([problem, n, k])+"/"+file+".csv")
    dfloc = df.loc[(df["n"] == 3) & (df["k"] == 3)]
    idx = dfloc.loc[dfloc["expanded transitions"] == dfloc["expanded transitions"].min()].iloc[0]["idx"]
    print(idx)

    with open("experiments/results/"+filename([problem, n, k])+"/"+file+"/"+str(idx)+".json", "r") as f:
        info = json.load(f)
    return onnx.load("experiments/results/"+filename([problem, n, k])+"/"+file+"/"+str(idx)+".onnx"), info


def exp_test_generalization(problem, file, up_to, timeout=10*60):
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if n == 0 or solved[n-1][k] or k == 0 or solved[n][k-1]:
                env = DCSSolverEnv(problem, n+1, k+1, max_actions=max_actions[problem, n+1, k+1])
                agent, info = pick_agent(problem, 2, 2, file)

                print("Testing agent with", problem, n+1, k+1)
                results = test_onnx(agent, env, timeout=timeout)
                print("Done.", results["synthesis time(ms)"])
                results.update(info)
                df.append(results)
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True
    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, 2, 2])+"/Agent"+str(up_to)+".csv")


def exp_test_ra(problem, up_to, old=False, timeout="10m"):
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if n == 0 or solved[n-1][k] or k == 0 or solved[n][k-1]:
                df.append(test(problem, n+1, k+1, "r", timeout=timeout, old=old))
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True

    df = pd.DataFrame(df)
    file = "RAold"+str(up_to)+".csv" if old else "RA"+str(up_to)+".csv"
    df.to_csv("experiments/results/"+filename([problem, 2, 2])+"/"+file)


if __name__ == "__main__":
    for problem in ["AT", "BW", "TL", "DP", "TA"]:
        exp_test_ra(problem, 15, timeout="1s")
        exp_test_generalization(problem, "10m_0", 15, timeout=1)


