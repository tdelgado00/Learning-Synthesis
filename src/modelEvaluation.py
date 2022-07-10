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


def frontier_size_stats(problem, n, k, eps, ra_feature):
    env = DCSSolverEnv(problem, n, k, ra_feature, False)
    df = []
    for i in range(eps):
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            df.append({"step": steps, "ep": i+1, "frontier size": len(obs)})
            action = np.random.randint(len(obs))

            obs2, reward, done, info = env.step(action)

            obs = obs2
            steps += 1
    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, n, k])+"/frontiers"+("RA" if ra_feature else "")+".csv")

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
        states = get_random_states(DCSSolverEnv(problem, n, k, True, True, True, True, True, True, True))

        file = "experiments/results/" + filename([problem, n, k]) + "/states_prop.pkl"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(states, f)


def read_random_states(problem, n, k, file, info):
    with open(results_path(problem, n, k, file), "rb") as f:
        states = pickle.load(f)

    with open("labels/"+problem+".txt", "r") as f:
        nlabels = len(list(f))

    check = lambda p: p in info.keys() and info[p]

    ra = check("ra feature")
    labels = check("labels")
    state_labels = check("state labels")
    context = check("context features")
    je = check("je feature")
    nk = check("nk feature")
    p = check("prop features")

    context_idx = 3
    statelabels_idx = context_idx + 4
    labels_idx = statelabels_idx + nlabels
    base_idx = labels_idx + nlabels
    je_idx = base_idx + 12
    nk_idx = je_idx + 2
    p_idx = nk_idx + 2

    idx = []
    if ra:
        idx += list(range(3))
    if context:
        idx += list(range(context_idx, context_idx + 4))
    if state_labels:
        idx += list(range(statelabels_idx, statelabels_idx + nlabels))
    if labels:
        idx += list(range(labels_idx, labels_idx + nlabels))
    idx += list(range(base_idx, base_idx + 12))
    if je:
        idx += list(range(je_idx, je_idx + 2))
    if nk:
        idx += list(range(nk_idx, nk_idx + 2))
    if p:
        idx += list(range(p_idx, p_idx + 4))

    return [s[:, idx] for s in states]


def save_model_q_dfs(problem, n, k, file, states_file, agent_selector, selector_name="best"):
    idx = agent_selector(problem, file)
    path = agent_path(filename([problem, 2, 2]) + "/" + file, idx)

    random_states = read_random_states(problem, n, k, states_file, get_agent_info(path))

    df = get_agent_q_df(problem, path, random_states)

    df.to_csv("experiments/results/" + filename([problem, n, k]) + "/"+file+"/"+selector_name+".csv")


if __name__ == "__main__":
    #for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    #    print(problem)
    #    if problem != "CM":
    #        frontier_size_stats(problem, 3, 3, 5, True)
    #        frontier_size_stats(problem, 3, 3, 5, False)
    save_all_random_states(2, 2)
    #save_all_random_states(2, 2)

    #features_search(DCSSolverEnv("TA", 2, 2, True), 100000)
    #features_search(DCSSolverEnv("DP", 2, 2, True), 100000)
    #features_search(DCSSolverEnv("BW", 2, 2, True), 100000)
    #features_search(DCSSolverEnv("CM", 2, 2, True), 100000)

    #for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    #    print(problem)
    #    features_search(DCSSolverEnv(problem, 3, 3, True), 100000)
