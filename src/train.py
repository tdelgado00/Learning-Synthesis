import os
import pickle

import numpy as np
from stable_baselines3 import DQN
from agent import Agent
from environment import DCSSolverEnv
from util import filename, get_problem_data, feature_names

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor

import pandas as pd


from test import test_old

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

models = {
    "M": lambda eta, nn_size: MLPRegressor(
        hidden_layer_sizes=(nn_size,),
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=eta),
    "L": lambda eta: SGDRegressor(
        learning_rate="constant",
        eta0=eta
    ),
    "DQN": lambda eta, env: DQN("MultiInputPolicy", env=env, learning_rate=eta)
}


def experiment_file(problem, n, k, experiment, params):
    name = filename([experiment]+params)
    return "experiments/results/" + filename([problem, n, k]) + "/" + name + ".pkl"


def train_agent(problem, n, k, minutes, dir, eta=1e-6, epsilon=0.1, nnsize=20, copy_freq=200000):
    env = DCSSolverEnv(problem, n, k, max_actions=max_actions[problem, n, k])

    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir

    agent = Agent(models["M"](eta, nnsize), dir=dir)
    agent.train(env, minutes * 60, copy_freq=copy_freq, epsilon=epsilon)


def test_agents(problem, n, k, file, problems):
    with open("experiments/results/"+filename([problem, n, k])+"/random_states.pkl", "rb") as f:
        random_states = pickle.load(f)
    df = []
    dir = "experiments/results/"+filename([problem, n, k])+"/"+file
    files = os.listdir(dir)
    for i in range(len(files)):
        with open(dir+"/"+files[i], "rb") as f:
            agent, time, steps = pickle.load(f)
        avg_q = np.mean([np.max(agent.eval(s)) for s in random_states])
        coefs = eval_agents_coefs(agent, problem, n, k)
        for problem2, n2, k2 in problems:
            env = DCSSolverEnv(problem2, n2, k2, max_actions=max_actions[problem2, n2, k2])
            print("Testing", problem2, n2, k2, file)
            result = agent.test(env)
            if result == "timeout":
                result = {"problem": problem2, "n": n2, "k": k2}
            result["training time"] = time
            result["training steps"] = steps
            result["avg q"] = avg_q
            result["idx"] = i
            result.update(coefs)
            df.append(result)

    df = pd.DataFrame(df)
    #df.to_csv("experiments/results/"+filename([problem, n, k])+"/"+file+".csv")


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


def exp_nn_size(problem, n, k, minutes):
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
    values = agent.eval(actions)
    values = (values - np.mean(values)) / np.std(values)
    model = LinearRegression().fit(actions, values)

    coefs = {}
    for i in range(len(feature_names)):
        coefs[feature_names[i]] = model.coef_[i]
    return coefs


if __name__ == "__main__":
    for problem in ["AT", "BW", "TL", "DP", "TA", "CM"]:
        train_agent(problem, 2, 2, 60, "60m", copy_freq=50000, epsilon=0.1, eta=1e-5)
        test_agents(problem, 2, 2, "60m", [(problem, 2, 2), (problem, 3, 3)])
