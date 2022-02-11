import pickle

from agent import Agent

from environment import DCSSolverEnv
from train import models
from util import filename
import itertools

import numpy as np
import pandas as pd


def get_random_states(problem, n, k, eps, size, file):
    nn_size = 20
    eta = 1e-6

    env = DCSSolverEnv(problem, n, k, max_actions=10000)
    agent = Agent(models["M"](eta, nn_size))

    idxs = np.random.choice(range(10000), 100)

    states = []
    done = True
    obs = None
    for i in range(10000):
        if done:
            obs = env.reset()
        if i in idxs:
            states.append(obs)
        action = np.random.randint(len(obs))
        obs, reward, done, info = env.step(action)

    with open(file, "wb") as f:
        pickle.dump(states, f)


def print_features(features):
    print("controlable:", features[0])
    print("goal:", features[1])
    print("error:", features[2])
    print("none:", features[3])
    print("marcado:", features[4])
    print("deadlock:", features[5])
    print("uncontrollability:", features[6])
    print("unexplorability:", features[7])
    print("1/depth:", features[8])
    print("stateUnexplorability:", features[9])

def save_q_values():
    problem, n, k = "BW", 2, 2
    env = lambda a, test: DCSSolverEnv(problem, n, k, a, test, max_actions=1000)
    agent = Agent.load("agents/" + filename([problem, 2, 2]) + "/good.pkl", env, env)

    features = [0,    # controlable: binaria
                0,    # goal: binaria
                0,    # error: binaria
                0,    # none: binaria
                0,    # marcado: binaria
                0,    # deadlock: binaria
                0,    # uncontrollability: [0,1]
                0,    # unexplorability: [0,1]
                0.1,  # 1/depth: [0,1]
                1     # stateUnexplorability: [0,1]
                ]

    possible_features = itertools.product(*[
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0, 0.1, 0.25, 0.5, 0.75, 1],
        [0, 0.1, 0.25, 0.5, 0.75, 1],
        [0, 0.1, 0.25, 0.5, 0.75, 1],
        [0, 0.1, 0.25, 0.5, 0.75, 1]
    ])
    df = []
    for features in possible_features:
        df.append({
            "controllable": features[0],
            "goal": features[1],
            "error": features[2],
            "none": features[3],
            "marcado": features[4],
            "deadlock": features[5],
            "uncontrollability": features[6],
            "unexplorability": features[7],
            "1/depth": features[8],
            "stateUnexplorability": features[9],
            "value": np.round(-agent.eval([features])[0], 3)
        })

    #print_features(features)
    #print("Agent value:", np.round(-agent.eval([features])[0], 3))
    df = pd.DataFrame(df)
    df.to_csv("agents/" + filename([problem, 2, 2]) + "/good_values.csv")

def save_all_random_states():
    for problem, n, k in [(problem, 2, 2) for problem in ["AT", "TL", "TA", "BW", "DP", "CM"]]:
        get_random_states(problem, n, k, 3, 100, "experiments/results/"+filename([problem, n, k])+"/random_states.pkl")


if __name__ == "__main__":
    save_all_random_states()
