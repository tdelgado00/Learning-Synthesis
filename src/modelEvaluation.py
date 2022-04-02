import os
import pickle
import sys
import onnx

from onnxruntime import InferenceSession
from sklearn.linear_model import LinearRegression
from environment import DCSSolverEnv
from util import filename
import numpy as np
from util import *
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_random_states(problem, n, k, name, total=20000, sampled=2000):
    env = DCSSolverEnv(problem, n, k, True)

    idxs = np.random.choice(range(total), sampled)

    states = []
    done = True
    obs = None
    for i in range(total):
        if done:
            obs = env.reset()

        if i in idxs:
            states.append(np.copy(obs))

        action = np.random.randint(len(obs))
        obs, reward, done, info = env.step(action)

    file = "experiments/results/"+filename([problem, n, k])+"/"+name+".pkl"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(states, f)


def eval_VIF(actions, features):
    df = pd.DataFrame(actions, columns=features)
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(features))]
    return vif_data


def get_agent_q_df(path, states):
    actions = np.array([a for s in states for a in s])

    agent = onnx.load(path)
    sess = InferenceSession(agent.SerializeToString())

    values = sess.run(None, {'X': actions})[0]
    features = feature_names(get_agent_info(path)["ra feature"])

    df = {features[i]: actions[:, i] for i in range(len(features))}
    df["q"] = [v[0] for v in values]

    return pd.DataFrame(df)


def eval_agent_q(path, random_states):
    agent = onnx.load(path)
    sess = InferenceSession(agent.SerializeToString())
    return np.mean([np.max(sess.run(None, {'X': s})) for s in random_states])


if __name__ == "__main__":
    for problem in ["DP", "TL", "TA", "BW", "CM"]:
        print(problem)
        get_random_states(problem, 2, 2, "states_2_2")