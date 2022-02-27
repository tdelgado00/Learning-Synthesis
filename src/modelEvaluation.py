import json
import os
import pickle

from onnxruntime import InferenceSession

from agent import Agent

from environment import DCSSolverEnv
from src.train import get_agent
from util import filename
import itertools

import numpy as np
import pandas as pd

from util import feature_names


def get_random_states(problem, n, k, total, sampled):
    env = DCSSolverEnv(problem, n, k, max_actions=100000)

    idxs = np.random.choice(range(total), sampled)

    states = []
    done = True
    obs = None
    for i in range(total):
        if done:
            obs = env.reset()

        if i in idxs:
            states.append(obs)

        action = np.random.randint(len(obs))
        obs, reward, done, info = env.step(action)

    file = "experiments/results/"+filename([problem, n, k])+"/random_states2.pkl"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(states, f)


def save_q_values():
    problem, n, k = "BW", 2, 2
    env = DCSSolverEnv(problem, n, k, max_actions=1000)
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

def save_all_random_states(n, k):
    for problem in ["AT", "TL", "TA", "BW", "DP", "CM"]:
        get_random_states(problem, n, k, 10000, 500)


def eval_agent(agent, features):
    sess = InferenceSession(agent.SerializeToString())
    return sess.run(None, {'X': features})


if __name__ == "__main__":
    print(eval_agent(get_agent("AT", 2, 2, "10m_0")[0],
        [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0.5, 1]]
    ))

    print(eval_agent(get_agent("AT", 2, 2, "10m_0")[0],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    ))
    #save_all_random_states(2, 2)
    #save_all_random_states(3, 3)
