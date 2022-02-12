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
from test import test_old
from util import filename, get_problem_data, feature_names
max_actions = {}


def get_max_actions():
    r = get_problem_data("mono")["expandedTransitions"]
    for problem in ["AT", "TA", "DP", "TL", "BW", "CM"]:
        for n in range(1, 6):
            for k in range(1, 6):
                if (problem, n, k) in r.keys() and r[problem, n, k] == r[problem, n, k]:
                    max_actions[problem, n, k] = int(r[problem, n, k])
                else:
                    max_actions[problem, n, k] = 1000000


get_max_actions()


def experiment_file(problem, n, k, experiment, params):
    name = filename([experiment]+params)
    return "experiments/results/" + filename([problem, n, k]) + "/" + name + ".pkl"


def train_agent(problem, n, k, minutes, dir, eta=1e-6, epsilon=0.1, nnsize=20, copy_freq=200000):
    env = DCSSolverEnv(problem, n, k, max_actions=max_actions[problem, n, k])

    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir if dir is not None else None

    agent = Agent(eta=eta, nnsize=nnsize, epsilon=epsilon, dir=dir)
    agent.train(env, minutes * 60, copy_freq=copy_freq)

    return agent


def eval_agent_q(agent, random_states):
    sess = InferenceSession(agent.SerializeToString())
    return np.mean([np.max(sess.run(None, {'X': s})) for s in random_states])


def test_agents(problem, n, k, file, problems):
    with open("experiments/results/"+filename([problem, n, k])+"/random_states.pkl", "rb") as f:
        random_states = pickle.load(f)

    df = []
    dir = "experiments/results/"+filename([problem, n, k])+"/"+file
    for i in range(len(os.listdir(dir))//2):
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


def exp_iterations(problem, n, k):
    for it in range(5):
        train_agent(problem, n, k, 20, "variance_"+str(it))


def exp_eta(problem, n, k, minutes):
    for eta in [1e-3, 1e-4, 1e-5, 1e-6]:
        for eps in [0.05, 0.1, 0.2]:
            for it in range(3):
                train_agent(problem, n, k, minutes, "eta_"+str(eta), eta=eta, epsilon=0.1)


def exp_epsilon(problem, n, k, minutes):
    for eps in [0.01, 0.05, 0.1, 0.2, 0.33]:
        train_agent(problem, n, k, minutes, "epsilon_"+str(eps), epsilon=eps)
        test_agents(problem, n, k, "epsilon_"+str(eps), [(problem, 2, 2), (problem, 3, 3)])


def exp_nnsize(problem, n, k, minutes):
    for nnsize in [5, 10, 20, 50, 100]:
        train_agent(problem, n, k, minutes, "nnsize_"+str(nnsize), nnsize=nnsize)


def pick_agent(problem, n, k, file):
    df = pd.read_csv("experiments/results/"+filename([problem, n, k])+"/"+file+".csv")
    idx = df.groupby("idx").loc[(df["n"] == 3) & (df["k"] == 3)]["expanded transitions"].argmax()

    with open("experiments/results/"+filename([problem, n, k])+"/"+file+".pkl", "rb") as f:
        return pickle.load(f)[idx][0]


def exp_high_generalization(problem, file, up_to=4):
    df = []
    for n, k in [(x, x) for x in range(2, up_to+1)]:

        env = lambda a, test: DCSSolverEnv(problem, n, k, max_actions=max_actions[problem, n, k])

        agent = pick_agent(problem, n, k, file)

        print("Running Agent...")
        results = agent.test(env)

        print("Running RA...")
        ra_result = test_old(problem, n, k, "r")

        df.append({
            "total trans": max_actions[problem, n, k],
            "agent trans": results["expanded transitions"],
            "agent time": int(results["synthesis time(ms)"]),
            "ra trans": ra_result["expanded transitions"],
            "ra time": ra_result["synthesis time(ms)"]
        })
        print("Done.")

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, 2, 2])+"/high_generalization.csv")


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


if __name__ == "__main__":
    for problem in ["AT", "BW", "TL", "DP", "TA"]:
        for it in range(3):
            train_agent(problem, 2, 2, 3, "10m_"+str(it), copy_freq=2000, epsilon=0.1, eta=1e-5)
            test_agents(problem, 2, 2, "10m"+str(it), [(problem, 2, 2), (problem, 3, 3)])




