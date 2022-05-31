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


def best_generalization_agent(problem, file):
    df = pd.read_csv("experiments/results/"+filename([problem, 2, 2])+"/"+file+"/generalization_all.csv")
    max_idx = df["idx"].max()
    solved = [0 for i in range(max_idx+1)]
    expanded = [0 for i in range(max_idx+1)]
    for x, cant in dict(df["idx"].value_counts()).items():
        solved[x] = cant
    for x, cant in dict(df.groupby("idx")["expanded transitions"].sum()).items():
        expanded[x] = cant
    perf = [(solved[i], expanded[i], i) for i in range(max_idx+1)]
    return max(perf, key=lambda t: (t[0], -t[1], t[2]))[2]


def test_ra_nico(problem, n, k, timeout="30m"):
    command = ["timeout", timeout, "java", "-Xmx8g", "-classpath", "mtsaNico.jar",
               "MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking.ReadyAbstractionHeuristic",
               "-i", fsp_path(problem, n, k)]

    proc = subprocess.run(command,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        results = read_results(proc.stdout.split("\n"))

    results["algorithm"] = "RAsola"
    results["heuristic"] = "r"
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results, None


def test_ra(problem, n, k, timeout="30m"):
    command = ["timeout", timeout, "java", "-Xmx8g", "-classpath", "mtsa.jar",
               "ltsa.ui.LTSABatch", "-i", fsp_path(problem, n, k), "-c", "DirectedController", "-r"]

    proc = subprocess.run(command,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        results = read_results(proc.stdout.split("\n"))

    results["algorithm"] = "OpenSet RA"
    results["heuristic"] = "r"
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results, None


def test_agent(path, problem, n, k, timeout="30m", debug=False, use_nk_feature=False):
    command = ["timeout", timeout, "java", "-Xmx8g", "-classpath", "mtsa.jar",
               "MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking.FeatureBasedExplorationHeuristic",
               "-i", fsp_path(problem, n, k),
               "-m", path
               ]

    if debug:
        command += ["-d"]

    if path != "mock" and uses_feature(path, "ra feature"):
        command += ["-r"]

    if path != "mock" and uses_feature(path, "labels"):
        command += ["-l", "labels/"+problem+".txt"]
    else:
        command += ["-l", "mock"]

    if path != "mock" and uses_feature(path, "context features"):
        command += ["-c"]

    if path != "mock" and uses_feature(path, "state labels"):
        command += ["-s"]

    if path != "mock" and uses_feature(path, "je feature"):
        command += ["-j"]

    if path != "mock" and (uses_feature(path, "nk feature") or use_nk_feature):
        command += ["-n"]

    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        lines = proc.stdout.split("\n")[2:]
        try:
            i = list(map((lambda l: "ExpandedStates" in l), lines)).index(True)
            j = list(map((lambda l: "DirectedController" in l), lines)).index(True)
            if debug:
                debug = parse_java_debug(lines[:j])
                results = read_results(lines[i:])
            else:
                debug = None
                results = read_results(lines[i:])
        except BaseException as err:
            print(command)
            for line in lines:
                print(line)
            for line in proc.stderr.split("\n"):
                print(line)
            raise
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


def parse_java_debug(debug):
    debug = [l for l in debug if l != "--------------------------"]
    steps = []
    i = 0
    while i < len(debug):
        first = debug[i].split(" ")
        n, nfeatures = int(first[0]), int(first[1])
        actions = []
        features = []
        for j in range(n):
            line = debug[i+1+j].split(" ")
            actions.append(line[0])
            features.append([float(x) for x in line[1:nfeatures+1]])
        values = [float(x) for x in debug[i+1+n:i+1+2*n]]
        selected = float(debug[i+1+2*n])
        steps.append({"frontier size": n, "nfeatures": nfeatures, "actions": actions, "features": features, "values": values, "selected": selected})
        i += n*2 + 2
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
        info["avg q"] = eval_agent_q(path, read_random_states(problem, n, k, random_states_file, info))
        info["idx"] = i
        df.append(info)

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, n, k]) + "/" + file + "/" + "q.csv")


def test_all_ra(problem, up_to, timeout="10m", name="all_ra", func=test_ra):
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if n == 0 or solved[n - 1][k] or k == 0 or solved[n][k - 1]:
                print("Testing ra with", problem, n, k)
                df.append(func(problem, n + 1, k + 1, timeout=timeout)[0])
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True

    df = pd.DataFrame(df)
    file = filename([name, up_to]) + ".csv"
    df.to_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file)


def test_all_agent(problem, file, up_to, timeout="10m", name="all"):
    idx_agent = best_generalization_agent(problem, file)
    print("Testing all", problem, "with agent", idx_agent)
    path = agent_path(file, idx_agent)

    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                print("Testing agent with", problem, n+1, k+1)
                df.append(test_agent(path, problem, n + 1, k + 1, timeout=timeout, use_nk_feature=True)[0])
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + file + "/" + name + ".csv")


def test_all_random(problem, up_to, timeout="10m", name="all_random"):
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                print("Testing random with", problem, n+1, k+1)
                df.append(test_agent("mock", problem, n + 1, k + 1, timeout=timeout)[0])
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + name + ".csv")

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
    start = time.time()
    for i in range(eps):
        print("Testing random with", problem, n, k, i, "- Time: ", time.time()-start)
        r = test_agent("mock", problem, n, k, "10m")[0]
        r["idx"] = i
        df.append(r)
    df = pd.DataFrame(df)
    df.to_csv("experiments/results/"+filename([problem, n, k])+"/"+file)


def get_problem_labels(problem, eps=5):
    actions = set()
    for i in range(eps):
        actions.update({x for step in list(pd.DataFrame(test_agent("mock", problem, 2, 2, "10m", debug=True)[1])["actions"]) for x in step})

    def simplify(l):
        return "".join([c for c in l if c.isalpha()])

    return {simplify(l) for l in actions}


if __name__ == "__main__":
    pass
    #print(test_heuristic_python("DP", 3, 3, ra_feature_heuristic))
    #for problem in ["AT", "BW", "CM", "DP", "TA", "TL"]:
    #    test_all_agent(problem, "5mill_RA", 15, timeout="10m")
    #    test_all_agent(problem, "5mill_L", 15, timeout="10m")

    #for problem in ["AT", "BW", "CM", "DP", "TA", "TL"]:
    #    test_all_ra(problem, 15, timeout="10m", name="all_ra_sola", func=test_ra_nico)
