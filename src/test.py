import pickle
import subprocess
import time
import os

import onnx
import pandas as pd
from onnxruntime import InferenceSession

from environment import DCSSolverEnv
from modelEvaluation import eval_agent_q
from util import *


def test_ra(problem, n, k, timeout="30m", old=False):
    jar = "mtsaOld.jar" if old else "mtsa.jar"
    command = ["timeout", timeout, "java", "-Xmx8g", "-classpath", jar,
               "ltsa.ui.LTSABatch", "-i", fsp_path(problem, n, k), "-c", "DirectedController", "-r"]

    proc = subprocess.run(command,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        results = read_results(proc.stdout.split("\n"))

    results["algorithm"] = "old" if old else "new"
    results["heuristic"] = "r"
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results, None


def test_agent(path, problem, n, k, timeout="30m", labels_dir="mock", debug=False):
    command = ["timeout", timeout, "java", "-Xmx8g", "-classpath", "mtsa.jar",
               "MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking.FeatureBasedExplorationHeuristic",
               "-i", fsp_path(problem, n, k),
               "-m", path,
               "-l", labels_dir
               ]

    if debug:
        command += ["-d"]

    if uses_ra(path):
        command += ["-r"]

    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        lines = proc.stdout.split("\n")[2:]
        i = lines.index('Composition:DirectedController = DirectedController')
        if debug:
            debug = parse_java_debug(lines[:i])
            results = read_results(lines[i+6:])
        else:
            debug = None
            results = read_results(lines[i+6:])

    results["algorithm"] = "new"
    results["heuristic"] = path
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


# Not using this function, we test with onnx from Java
def test_onnx(path, problem, n, k, timeout=30 * 60, debug=None):
    with open(path[:-5] + ".json", "r") as f:
        info = json.load(f)

    env = DCSSolverEnv(problem, n, k, info["ra feature"])

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


def test_agents(problem, n, k, problem2, n2, k2, file, freq=1):
    df = []

    dir = results_path(problem, n, k, file)
    files = [f for f in os.listdir(dir) if f.endswith(".onnx")]
    for i in range(0, len(files), freq):
        print("Testing", i, "with", problem2, n2, k2)
        path = agent_path(problem, n, k, file, i)
        result, debug = test_agent(path, problem2, n2, k2, timeout="10m")

        if result == "timeout":
            result = {"problem": problem2, "n": n2, "k": k2}
        result.update(get_agent_info(path))
        result["idx"] = i
        df.append(result)

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, n, k]) + "/" + file + "/" + filename([problem2, n2, k2]) + ".csv")


def get_correct_features(states, info):
    if not info["ra feature"]:
        return [s[:, 2:] for s in states]
    else:
        return states


def test_agents_q(problem, n, k, file, random_states_file, freq=1):
    with open(results_path(problem, n, k, random_states_file), "rb") as f:
        random_states = pickle.load(f)
    df = []

    dir = results_path(problem, n, k, file)
    files = [f for f in os.listdir(dir) if f.endswith(".onnx")]
    for i in range(0, len(files), freq):
        print("Testing q", i)
        path = agent_path(problem, n, k, file, i)
        info = get_agent_info(path)
        info["avg q"] = eval_agent_q(path, get_correct_features(random_states, info))
        info["idx"] = i
        df.append(info)

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, n, k]) + "/" + file + "/" + "q.csv")


def test_all_ra(problem, up_to, old=False, timeout="10m", name="all_ra"):
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if n == 0 or solved[n - 1][k] or k == 0 or solved[n][k - 1]:
                print("Testing ra with", problem, n, k, "- Old:", old)
                df.append(test_ra(problem, n + 1, k + 1, timeout=timeout, old=old)[0])
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True

    df = pd.DataFrame(df)
    file = filename([name, up_to]) + (".csv" if not old else "_old.csv")
    df.to_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file)