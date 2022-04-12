import os
import pickle
import sys
import onnx

from onnxruntime import InferenceSession
from environment import DCSSolverEnv
from util import *
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_random_experience(env, total):

    states = []
    done = True
    obs = None
    steps = 0
    for i in range(total):
        if done:
            obs = env.reset()

        action = np.random.randint(len(obs))

        obs2, reward, done, info = env.step(action)

        states.append((obs,  action, 0 if done else -1, obs2, -steps-1))
        obs = obs2
        steps += 1

    return states


def get_random_states(env, total=20000, sampled=2000):
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

    return states


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


def save_all_random_states():
    for problem in ["AT", "DP", "TL", "TA", "BW", "CM"]:
        print(problem)
        states = get_random_states(DCSSolverEnv(problem, 2, 2, True))

        file = "experiments/results/" + filename([problem, n, k]) + "/states_no_conflict.pkl"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(states, f)


def read_random_states(problem, n, k, file, ra_feature):
    with open(results_path(problem, n, k, file), "rb") as f:
        states = pickle.load(f)

    if not ra_feature:
        return [s[:, 2:] for s in states]
    else:
        return states

def save_models_q_dfs(last=False):

    problems = ["AT", "TA", "TL", "DP", "BW", "CM"]
    for problem, n, k in [(x, 2, 2) for x in problems]:
        problem2, n2, k2 = (problem, 3, 3) if problem != "CM" else (problem, 2, 2)
        df1 = pd.read_csv("experiments/results/" + filename([problem, n, k]) + "/ra_feature_2h/" + filename(
            [problem2, n2, k2]) + ".csv")
        df2 = pd.read_csv("experiments/results/" + filename([problem, n, k]) + "/base_features_2h/" + filename(
            [problem, n2, k2]) + ".csv")
        idx_ra = best_agent_idx(df1) if not last else last_agent_idx(df1)
        idx_bf = best_agent_idx(df2) if not last else last_agent_idx(df2)

        random_states_ra = read_random_states(problem, n, k, "states_2_2.pkl", True)
        random_states_bf = read_random_states(problem, n, k, "states_2_2.pkl", False)

        df_ra = get_agent_q_df(agent_path(problem, n, k, "ra_feature_2h", idx_ra), random_states_ra)
        df_bf = get_agent_q_df(agent_path(problem, n, k, "base_features_2h", idx_bf), random_states_bf)

        t = "last" if last else "best"
        df_ra.to_csv("experiments/results/" + filename([problem, n, k]) + "/ra_feature_2h/"+t+"_"+str(idx_ra)+".csv")
        df_bf.to_csv("experiments/results/" + filename([problem, n, k]) + "/base_features_2h/"+t+"_"+str(idx_bf)+".csv")


if __name__ == "__main__":
    #save_models_q_dfs(last=False)
    #save_models_q_dfs(last=True)
    save_all_random_states()