import os
import pickle
from onnxruntime import InferenceSession
from sklearn.linear_model import LinearRegression
from environment import DCSSolverEnv
from util import filename
import numpy as np
from util import feature_names


def get_random_states(problem, n, k, total, sampled, name):
    env = DCSSolverEnv(problem, n, k)

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

    file = "experiments/results/"+filename([problem, n, k])+"/"+name+".pkl"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(states, f)


def save_all_random_states(n, k, name):
    for problem in ["AT", "TL", "TA", "BW", "DP", "CM"]:
        get_random_states(problem, n, k, 10000, 500, name)


def eval_agent(agent, features):
    sess = InferenceSession(agent.SerializeToString())
    return sess.run(None, {'X': features})


def eval_agents_coefs(agent, states):
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


if __name__ == "__main__":
    save_all_random_states(2, 2, "ra_feature_states")
