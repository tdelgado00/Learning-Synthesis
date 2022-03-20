import json
import subprocess
import time

import numpy as np
import onnx
import pandas as pd
from onnxruntime import InferenceSession

from src.environment import DCSSolverEnv
from util import read_results, filename


def test(problem, n, k, heuristic, timeout="30m", old=False, agent_dir="10m_0", agent_idx=0, labels_dir="mock",
         debug=False):
    if old:
        jar = "mtsaOld.jar"
    else:
        jar = "mtsa.jar"
    path = "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp"
    command = ["timeout", timeout, "java", "-Xmx8g", "-classpath", jar,
               "ltsa.ui.LTSABatch", "-i", path, "-c", "DirectedController", "-" + heuristic]
    if heuristic == "e":
        command += ["-p", "experiments/results/" + filename([problem, 2, 2]) + "/" + agent_dir + "/" + str(
            agent_idx) + ".onnx"]
        command += ["-l", labels_dir]
        if debug:
            command += ["-a"]
    proc = subprocess.run(command,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        lines = proc.stdout.split("\n")
        if debug:
            debug = parse_java_debug(lines[:-10])
            results = read_results(lines[-10:])
        else:
            debug = None
            results = read_results(lines[-10:])
    results["algorithm"] = "old" if old else "new"
    results["heuristic"] = heuristic
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results, debug


def test_mono(problem, n, k):
    proc = subprocess.Popen(
        ["timeout", "30m", "java", "-cp", "mtsaOld.jar", "ltsa.ui.LTSABatch", "-i",
         "fsp/" + problem + "/" + problem + "-" + str(n) + "-" + str(k) + ".fsp", "-c", "MonolithicController"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    results = read_results(proc.stdout.split("\n"))
    results["algorithm"] = "mono"
    results["heuristic"] = None
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results


def test_onnx(model, env, timeout=30 * 60, debug=None):
    start_time = time.time()
    sess = InferenceSession(model.SerializeToString())

    obs = env.reset()
    done = False
    info = None

    debug = None if not debug else []

    while not done and time.time() - start_time < timeout:
        values = sess.run(None, {'X': obs})
        action = np.argmax(values)
        if debug is not None:
            debug.append({"features": [[f for f in a] for a in obs], "values": [v[0] for v in values[0]]})
        obs, reward, done, info = env.step(action)

    return (info if time.time() - start_time < timeout else {
        "problem": env.problem,
        "n": env.n,
        "k": env.k,
        "synthesis time(ms)": np.nan,
        "expanded transitions": np.nan
    }), debug


def parse_java_debug(debug):
    debug = " ".join(debug).split("--------------------------")
    steps = []
    for step in filter(lambda x: x != "", debug):
        step = step.split(" ")
        step = list(filter(lambda x: x != "", step))
        n, nfeatures = int(step[0]), int(step[1])
        step = step[2:]
        features = []
        actions = []
        for i in range(n):
            actions.append(step[0])
            features.append([float(x) for x in step[1:nfeatures + 1]])
            step = step[nfeatures + 1:]
        values = [float(x) for x in step[:n]]
        selected = int(step[n])
        steps.append(
            {"frontier size": n, "nfeatures": nfeatures, "actoins": actions, "features": features, "values": values,
             "selected": selected})
    return steps


def pick_agent(problem, n, k, file):
    df = pd.read_csv("experiments/results/" + filename([problem, n, k]) + "/" + file + ".csv")
    dfloc = df.loc[(df["n"] == 3) & (df["k"] == 3)]
    idx = dfloc.loc[dfloc["expanded transitions"] == dfloc["expanded transitions"].min()].iloc[0]["idx"]
    return idx


def get_agent(problem, n, k, file, idx=None):
    if idx is None:
        idx = pick_agent(problem, n, k, file)
    with open("experiments/results/" + filename([problem, n, k]) + "/" + file + "/" + str(idx) + ".json", "r") as f:
        info = json.load(f)
    return onnx.load("experiments/results/" + filename([problem, n, k]) + "/" + file + "/" + str(idx) + ".onnx"), info


def test_from_python(problem, n, k, agent_dir, agent_idx, debug=None):
    env = DCSSolverEnv(problem, n, k)
    agent = get_agent(problem, 2, 2, agent_dir, agent_idx)[0]
    return test_onnx(agent, env, debug=debug)


def testJavaAndPythonCoherent():
    for problem, n, k, agent_dir, agent_idx in [("AT", 2, 2, "ra_feature", 8), ("AT", 3, 3, "ra_feature", 8)]:
        result, debug_java = test(problem, n, k, "e", agent_dir=agent_dir, agent_idx=agent_idx, debug=True)
        result, debug_python = test_from_python(problem, n, k, agent_dir=agent_dir, agent_idx=agent_idx, debug=True)
        assert len(debug_java) == len(debug_python)
        for i in range(len(debug_java)):
            if not np.allclose(debug_python[i]["features"], debug_java[i]["features"]):
                print(i)
                print(debug_python[i]["features"], debug_java[i]["features"])
            if not np.allclose(debug_python[i]["values"], debug_java[i]["values"]):
                print(i)
                print(debug_python[i]["features"], debug_java[i]["features"])



if __name__ == '__main__':
    testJavaAndPythonCoherent()
