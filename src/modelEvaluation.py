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


def features_search(env, total):
    done = True
    obs = None
    steps = 0
    count = 0
    for i in range(total):
        if done:
            obs = env.reset()

        if np.any([obs[i][9] > 0.5 for i in range(len(obs))]):
            print("Found child goal", steps)
            count += 1

        action = np.random.randint(len(obs))

        obs2, reward, done, info = env.step(action)

        obs = obs2
        steps += 1
    print("Total:", count)

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


def get_agent_q_df(problem, path, states):
    actions = np.array([a for s in states for a in s])

    agent = onnx.load(path)
    sess = InferenceSession(agent.SerializeToString())

    values = sess.run(None, {'X': actions})[0]
    features = feature_names(get_agent_info(path), problem)

    df = {features[i]: actions[:, i] for i in range(len(features))}
    df["q"] = [v[0] for v in values]

    return pd.DataFrame(df)


def eval_agent_q(path, random_states):
    agent = onnx.load(path)
    sess = InferenceSession(agent.SerializeToString())
    return np.mean([np.max(sess.run(None, {'X': s})) for s in random_states])


def save_all_random_states(n, k):
    for problem in ["AT", "DP", "TL", "TA", "BW", "CM"]:
        print(problem)
        states = get_random_states(DCSSolverEnv(problem, n, k, True, True))

        file = "experiments/results/" + filename([problem, n, k]) + "/states_labels.pkl"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(states, f)


def read_random_states(problem, n, k, file, info):
    with open(results_path(problem, n, k, file), "rb") as f:
        states = pickle.load(f)
    
    if not info["ra feature"] and not info["labels"]:
        return [s[:, -12:] for s in states]
    elif info["ra feature"] and not info["labels"]:
        return [s[:, :2]+s[:, -12:] for s in states]
    elif not info["ra feature"] and info["labels"]:
        return [s[2:] for s in states]
    else:
        return states


def save_model_q_dfs(problem, n, k, file, states_file, last=False):
    problem2, n2, k2 = problem, 3, 3
    df = pd.read_csv("experiments/results/" + filename([problem, n, k]) + "/"+file+"/" + filename(
        [problem2, n2, k2]) + ".csv")
    idx = best_agent_idx(df) if not last else last_agent_idx(df)

    random_states = read_random_states(problem, n, k, states_file, {"ra feature": True, "labels": True})

    df_ra = get_agent_q_df(problem, agent_path(problem, n, k, file, idx), random_states)

    t = "last" if last else "best"
    df_ra.to_csv("experiments/results/" + filename([problem, n, k]) + "/"+file+"/"+t+"_"+str(idx)+".csv")


if __name__ == "__main__":
    save_all_random_states(2, 2)

    #features_search(DCSSolverEnv("TA", 2, 2, True), 100000)
    #features_search(DCSSolverEnv("DP", 2, 2, True), 100000)
    #features_search(DCSSolverEnv("BW", 2, 2, True), 100000)
    #features_search(DCSSolverEnv("CM", 2, 2, True), 100000)

    #for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    #    print(problem)
    #    features_search(DCSSolverEnv(problem, 3, 3, True), 100000)
