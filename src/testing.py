import onnx
import os
from onnxruntime import InferenceSession
from environment import DCSSolverEnv
from util import *
import numpy as np
import subprocess
import time


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


def test_agent(path, problem, n, k, max_frontier=1000000, timeout="30m", debug=False):
    command = ["timeout", timeout, "java", "-Xmx8g", "-XX:MaxDirectMemorySize=512m", "-classpath", "mtsa.jar",
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

    state_labels_info = get_agent_info(path).get("state labels")
    if path != "mock" and state_labels_info is not None:
        if state_labels_info == True:
            command += ["-s", "1"]
        else:
            command += ["-s", str(state_labels_info)]

    if path != "mock" and uses_feature(path, "je feature"):
        command += ["-j"]

    if path != "mock" and uses_feature(path, "nk feature"):
        command += ["-n"]

    if path != "mock" and uses_feature(path, "prop feature"):
        command += ["-p"]

    if path != "mock" and uses_feature(path, "visits feature"):
        command += ["-v"]

    if path != "mock" and uses_feature(path, "only boolean"):
        command += ["-b"]

    command += ["-f", str(max_frontier)]

    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False}
    else:
        lines = proc.stdout.split("\n")[2:]
        err_lines = proc.stderr.split("\n")
        if np.any(["OutOfMem" in line for line in err_lines]):
            debug = None
            results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": True}
        else:
            try:
                i = list(map((lambda l: "ExpandedStates" in l), lines)).index(True)
                j = list(map((lambda l: "DirectedController" in l), lines)).index(True)
                if debug:
                    debug = parse_java_debug(lines[:j])
                    results = read_results(lines[i:])
                else:
                    debug = None
                    results = read_results(lines[i:])
                results["OutOfMem"] = False
            except BaseException as err:
                results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False, "Exception": True}
                print("Exeption!")
                print(" ".join(command))
                if np.any([("Frontier" in line) for line in err_lines]):
                    print("Frontier did not fit in the buffer.")
                else:
                    for line in lines:
                        print(line)
                    for line in err_lines:
                        print(line)
            
    results["algorithm"] = "new"
    results["heuristic"] = path
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results, debug


def test_monolithic(problem, n, k):
    proc = subprocess.Popen(
        ["timeout", "30m", "java", "-cp", "mtsaOld.jar", "ltsa.ui.LTSABatch", "-i",
         fsp_path(problem, n, k), "-c", "MonolithicController"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    results = read_results(proc.stdout.split("\n"))
    results["algorithm"] = "mono"
    results["heuristic"] = None
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results


def test_random(problem, n, k, times, file):
    df = []
    start = time.time()
    for i in range(times):
        print("Testing random with", problem, n, k, i, "- Time: ", time.time() - start)
        r = test_agent("mock", problem, n, k, timeout="10m")[0]
        r["idx"] = i
        df.append(r)
    df = pd.DataFrame(df)
    df.to_csv(results_path(problem, n, k, file))


def parse_java_debug(debug):
    """ This function parses the output of the synthesis procedure when -d flag is used """
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


def test_ra_all_instances(problem, up_to, timeout="10m", name="all_ra", func=test_ra):
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
    df.to_csv(results_path(problem, 2, 2, file))


def test_agent_all_instances(problem, file, up_to, timeout="10m", name="all", selection=None, max_frontier=1000000):
    idx_agent = selection(problem, file)
    print("Testing all", problem, "with agent", idx_agent)
    path = agent_path(problem, file, idx_agent)
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                print("Testing agent with", problem, n+1, k+1)
                df.append(test_agent(path, problem, n + 1, k + 1, max_frontier=max_frontier, timeout=timeout)[0])
                if not np.isnan(df[-1]["synthesis time(ms)"]):
                    solved[n][k] = True

    df = pd.DataFrame(df)
    df.to_csv(results_path(problem, 2, 2, file) + "/" + name + ".csv")


def test_random_all_instances(problem, up_to, timeout="10m", name="all_random"):
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
    df.to_csv(results_path(problem, 2, 2) + name + ".csv")


def test_training_agents_generalization(problem, file, up_to, timeout, total=100, max_frontier=1000000):
    df = []
    start = time.time()
    agents_saved = sorted([int(f[:-5]) for f in os.listdir(results_path(problem, file=file)) if "onnx" in f])
    np.random.seed(0)
    tested_agents = sorted(np.random.choice(agents_saved, min(total, len(agents_saved)), replace=False))
    for i in tested_agents:
        path = agent_path(problem, file, i)

        solved = [[False for _ in range(up_to)] for _ in range(up_to)]
        print("Testing agent", i, "with 5s timeout. Time:", time.time() - start)
        for n in range(up_to):
            for k in range(up_to):
                if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                    df.append(test_agent(path, problem, n + 1, k + 1, max_frontier=max_frontier, timeout=timeout)[0])
                    df[-1]["idx"] = i
                    if not np.isnan(df[-1]["synthesis time(ms)"]):
                        solved[n][k] = True
        print("Solved:", np.sum(solved))

    df = pd.DataFrame(df)
    df.to_csv(results_path(problem, 2, 2, file) + "/generalization_all.csv")


def test_onnx(path, problem, n, k, timeout=30 * 60, debug=None):
    """ Not using this function, we test with onnx from Java """

    with open(path[:-5] + ".json", "r") as f:
        info = json.load(f)

    env = DCSSolverEnv(problem, n, k, info)

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
            debug.append(
                {"features": [[f for f in a] for a in obs], "values": [v[0] for v in values[0]], "selected": action})
        obs, reward, done, info = env.step(action)

    return (info if time.time() - start_time < timeout else {
        "problem": env.problem,
        "n": env.n,
        "k": env.k,
        "synthesis time(ms)": np.nan,
        "expanded transitions": np.nan
    }), debug


def ra_feature_heuristic(obs, verbose=True):
    """ A handcrafted heuristic that attempts to emulate RA using the RA features """
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
    return max_i


def heuristic_test_exp(problem, n, k, times, heuristic, features, file, verbose=False):
    """ Function used for testing a custom heuristic """

    def test_heuristic_python():
        env = DCSSolverEnv(problem, n, k, features)

        obs = env.reset()
        done = False
        info = None
        while not done:
            if verbose:
                print("---------------------------")
            action = heuristic(obs, verbose)
            obs, reward, done, info = env.step(action)

        return info, None

    df = []
    for i in range(times):
        r = test_heuristic_python()[0]
        r["idx"] = i
        df.append(r)
    return df


def get_problem_labels(problem, eps=5):
    """ Used to generate the set of labels for a given parametric problem """
    actions = set()
    for i in range(eps):
        actions.update({x for step in
                        list(pd.DataFrame(test_agent("mock", problem, 2, 2, timeout="10m", debug=True)[1])["actions"])
                        for x in step})

    def simplify(l):
        return "".join([c for c in l if c.isalpha()])

    return {simplify(l) for l in actions}


if __name__ == '__main__':
    pass
