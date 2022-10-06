import onnx, os

from onnxruntime import InferenceSession
from environment import DCSSolverEnv
from util import *
import pandas as pd


def save_all_random_states(n, k, features):
    """ Saves observations from a random policy for all problems in the benchmark """
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

    for problem in ["AT", "DP", "TL", "TA", "BW", "CM"]:
        print("Saving random states for problem", problem)
        states = get_random_states(DCSSolverEnv(problem, n, k, features))

        file = results_path(problem, n, k, "states_b.pkl")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(states, f)


def read_random_states(problem, n, k, file, idx_features=None):
    """ Reads random states for a given problem, selecting only features of idx_features if given """
    with open(results_path(problem, n, k, file), "rb") as f:
        states = pickle.load(f)

    if idx_features is None:
        return states
    else:
        return [s[:, idx_features] for s in states]


def save_model_q_df(problem, n, k, file, idx, states_file, idx_features=None, name="features_q"):
    """ Saves a dataframe with feature to network output associations """
    path = agent_path(problem, file, idx)

    random_states = read_random_states(problem, n, k, states_file, idx_features=idx_features)

    actions = np.array([a for s in random_states for a in s])

    agent = onnx.load(path)
    sess = InferenceSession(agent.SerializeToString())

    values = sess.run(None, {'X': actions})[0]
    features = feature_names(get_agent_info(path), problem)

    df = {features[i]: actions[:, i] for i in range(len(features))}
    df["q"] = [v[0] for v in values]

    df = pd.DataFrame(df)

    df.to_csv(results_path(problem, n, k, file) + "/" + name + ".csv")


def test_agents_q(problem, n, k, file, random_states_file, idx_features=None, freq=1):
    """ Testing Q for a set of random states for all models trained in file """

    def eval_agent_q(path, random_states):
        agent = onnx.load(path)
        sess = InferenceSession(agent.SerializeToString())
        return np.mean([np.max(sess.run(None, {'X': s})) for s in random_states])

    df = []
    path = results_path(problem, n, k, file)
    files = [f for f in os.listdir(path) if f.endswith(".onnx")]
    for i in range(0, len(files), freq):
        a_path = agent_path(problem, file, i)
        info = get_agent_info(a_path)
        info["avg q"] = eval_agent_q(a_path, read_random_states(problem, n, k, random_states_file, idx_features))
        info["idx"] = i
        df.append(info)

    df = pd.DataFrame(df)
    df.to_csv(results_path(problem, n, k, file) + "/" + "q.csv")