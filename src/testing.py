import json
import time

import onnx
import pandas as pd
from onnxruntime import InferenceSession

from src.environment import DCSSolverEnv
from src.util import filename
from test import test_agent, agent_path
from train import train_agent
import numpy as np


def test_java_and_python_coherent():
    print("Testing java and python coherent")
    for problem, n, k, agent_dir, agent_idx in [("AT", 2, 2, "testing", 1)]:
        for test in range(10):
            print("Test", test)
            result, debug_java = test_agent(agent_path(filename([problem, 2, 2])+"/"+"testing", 1), problem, n, k, debug=True)
            result, debug_python = test_onnx(agent_path(filename([problem, 2, 2])+"/"+"testing", 1), problem, n, k, debug=True)
            assert len(debug_java) == len(debug_python)
            for i in range(len(debug_java)):
                if len(debug_python[i]["features"]) != len(debug_java[i]["features"]):
                    print("Different frontier size!", len(debug_python[i]["features"]), len(debug_java[i]["features"]), "at step", i)
                for j in range(len(debug_python[i]["features"])):
                    if not np.allclose(debug_python[i]["features"][j], debug_java[i]["features"][j]):
                        print(i, j, "Features are different")
                        print("Python", list(debug_python[i]["features"][j]))
                        print("Java", debug_java[i]["features"][j])
                    if not np.allclose(debug_python[i]["values"][j], debug_java[i]["values"][j]):
                        print(i, j, "Values are different")
                        print("Python", debug_python[i]["values"][j])
                        print("Java", debug_java[i]["values"][j])


def test_train_agent():
    print("Training agent")
    start = time.time()
    train_agent("AT", 2, 2, "testing", max_steps=1000, copy_freq=500, ra_feature=True, labels=True, context_features=True,
                state_labels=True, je_feature=True, experience_replay=True, fixed_q_target=True)
    _, _ = test_agent(agent_path(filename(["AT", 2, 2])+"/"+"testing", 1), "AT", 2, 2, debug=False)
    print(time.time() - start)


def test_target_and_buffer():
    problem, n, k = "AT", 2, 2
    max_steps = 100
    copy_freq = 100
    buffer_size = 5
    batch_size = 3
    reset_target = 10

    train_agent(problem, n, k, None, max_steps=max_steps, copy_freq=copy_freq, ra_feature=True, labels=True,
                context_features=True, je_feature=True,
                fixed_q_target=True, reset_target_freq=reset_target,
                experience_replay=True, buffer_size=buffer_size, batch_size=batch_size, verbose=True)
    #test_agents(problem, n, k, problem, n, k, file)
    #test_agents_q(problem, n, k, file, "states_no_conflict.pkl")


def tests():
    test_train_agent()
    test_java_and_python_coherent()

# Not using this function, we test with onnx from Java
def test_onnx(path, problem, n, k, timeout=30 * 60, debug=None):
    with open(path[:-5] + ".json", "r") as f:
        info = json.load(f)

    env = DCSSolverEnv(problem, n, k, info["ra feature"], info["labels"], info["context features"], info["state labels"], info["je feature"])

    agent = onnx.load(path)

    start_time = time.time()
    sess = InferenceSession(agent.SerializeToString())

    obs = env.reset()
    done = False
    info = None

    debug = None if not debug else []

    while not done and time.time() - start_time < timeout:
        values = sess.run(None, {'X': obs})
        action = np.argmax(values)
        if debug is not None:
            debug.append({"features": [[f for f in a] for a in obs], "values": [v[0] for v in values[0]], "selected": action})
        obs, reward, done, info = env.step(action)

    return (info if time.time() - start_time < timeout else {
        "problem": env.problem,
        "n": env.n,
        "k": env.k,
        "synthesis time(ms)": np.nan,
        "expanded transitions": np.nan
    }), debug


def test_heuristic_python(problem, n, k, heuristic, verbose=False):
    env = DCSSolverEnv(problem, n, k, True)

    obs = env.reset()
    done = False
    info = None
    c = 0
    while not done:
        if verbose:
            print("---------------------------")
        action, count = heuristic(obs, verbose)
        c += count
        obs, reward, done, info = env.step(action)

    print("Desempates por depth:", c)
    return info, None


def heuristic_test_exp(problem, n, k, eps, heuristic, file, verbose=False):
    df = []
    for i in range(eps):
        r = test_heuristic_python(problem, n, k, heuristic, verbose)[0]
        r["idx"] = i
        df.append(r)
    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, n, k])+"/"+file)

if __name__ == '__main__':
    tests()
