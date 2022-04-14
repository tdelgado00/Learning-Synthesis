import pickle
import subprocess
import time
import os

import onnx
import pandas as pd
from onnxruntime import InferenceSession

from environment import DCSSolverEnv
from modelEvaluation import eval_agent_q, read_random_states
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

    if path != "mock" and uses_ra(path):
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



def test_agents_q(problem, n, k, file, random_states_file, freq=1):
    df = []

    dir = results_path(problem, n, k, file)
    files = [f for f in os.listdir(dir) if f.endswith(".onnx")]
    for i in range(0, len(files), freq):
        print("Testing q", i)
        path = agent_path(problem, n, k, file, i)
        info = get_agent_info(path)
        info["avg q"] = eval_agent_q(path, read_random_states(problem, n, k, random_states_file, info["ra feature"]))
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


def random_heuristic(obs):
    return np.random.randint(0, obs.shape[0])


def ra_feature_heuristic(obs, verbose=True):
    max_i = -1
    count = 0
    for i in range(0, len(obs)):
        if verbose:
            print(list(np.round(obs[i], 2)))
        obs[i][1] *= -1
        if obs[i][2] < 0.5: # not in open
            if verbose:
                print("not in open")
            continue
        if max_i == -1:
            if verbose:
                print("first possibility")
            max_i = i
        elif obs[i][3] < obs[max_i][3]:
            if verbose:
                print("better controllability")
            max_i = i
        elif np.isclose(obs[i][3], obs[max_i][3]):
            if obs[i][3] < 0.5:  # uncontrollable
                obs[i][0:2] *= -1
            dist, best_dist = tuple(obs[i][0:2]), tuple(obs[max_i][0:2])

            if dist < best_dist:
                if verbose:
                    print("better distance")
                max_i = i
            elif dist == best_dist and obs[i][4] > obs[max_i][4]:
                if verbose:
                    print("same distance, better depth")
                count += 1
                max_i = i

    if verbose:
        print(max_i)
    assert max_i != -1
    return max_i, count


def heuristic_test_exp(problem, n, k, eps, heuristic, file, verbose=False):
    df = []
    for i in range(eps):
        r = test_heuristic_python(problem, n, k, heuristic, verbose)[0]
        r["idx"] = i
        df.append(r)
    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, n, k])+"/"+file)


def test_random_exp(problem, n, k, eps, file):
    df = []
    for i in range(eps):
        r = test_agent("mock", problem, n, k, "10m")[0]
        r["idx"] = i
        df.append(r)
    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, n, k])+"/"+file)


if __name__ == "__main__":
    for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
        test_random_exp(problem, 2, 2, 200, "random.csv")
        test_random_exp(problem, 3, 3, 20, "random.csv")

    #for problem in ["AT", "TA", "TL", "DP", "BW", "CM"]:
    #    n, k = 2, 2
    #    print(problem)
    #    print(test_heuristic_python(problem, n, k, ra_feature_heuristic, verbose=False)[0]["expanded transitions"])
    #    print(test_ra(problem, n, k)[0]["expanded transitions"])