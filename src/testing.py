from util import *
import numpy as np
import subprocess
import time
import os
from util import parse_java_debug


def test_ra(problem, n, k, timeout="30m", ebudget=-1):
    """Testing RA in a given instance of a problem"""
    command = ["timeout", timeout, "java", "-Xmx8g", "-classpath", "mtsa.jar",
               "ltsa.ui.LTSABatch", "-i", fsp_path(problem, n, k), "-c", "DirectedController", "-r"]
    if ebudget > -1:
        command.append("-e")
        command.append(str(ebudget))

    print("Testing RA at ", str(n), " ", str(k), " of ", problem)
    start = time.time()
    proc = subprocess.run(command,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    end = time.time()
    print("tested in ", end - start, " seconds")

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan}
    else:
        results = read_results(proc.stdout.split("\n"))

    expansionMsg = ""
    if (ebudget > -1 and results["expanded transitions"] > ebudget):
        expansionMsg = ", EXPANSION BUDGET EXCEEDED"
    print("Total expanded transitions: ", results["expanded transitions"], expansionMsg)
    results["algorithm"] = "OpenSet RA"
    results["heuristic"] = "r"
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    return results, None


def test_agent(path, problem, n, k, max_frontier=1000000, timeout="30m", debug=False, ebudget=-1,
               file="not specified in test_agent", verbose=False, components_by_state = False):
    """Testing a specific agent in a given instance of a problem"""
    command = ["timeout", timeout, "java", "-Xmx8g", "-XX:MaxDirectMemorySize=512m", "-classpath", "mtsa.jar",
               "MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking.FeatureBasedExplorationHeuristic",
               "-i", fsp_path(problem, n, k),
               "-m", path]

    if debug:
        command += ["-d"]

    if path != "mock" and uses_feature(path, "ra feature"):
        command += ["-r"]

    if path != "mock" and uses_feature(path, "labels"):
        command += ["-l", "labels/" + problem + ".txt"]
    else:
        command += ["-l", "mock"]

    if path != "mock" and uses_feature(path, "context features"):
        command += ["-c"]

    if path != "mock":
        state_labels_info = get_agent_info(path).get("state labels")
        if state_labels_info is not None:
            if state_labels_info:
                command += ["-s", "1"]
            else:
                command += ["-s", str(state_labels_info)]
    else:
        command += ["-s", "0"]

    if path != "mock" and uses_feature(path, "je feature"): command += ["-j"]

    if path != "mock" and uses_feature(path, "nk feature"): command += ["-n"]

    if path != "mock" and uses_feature(path, "prop feature"): command += ["-p"]

    if path != "mock" and uses_feature(path, "visits feature"): command += ["-v"]

    if path != "mock" and uses_feature(path, "only boolean"): command += ["-b"]

    if path != "mock" and uses_feature(path, "labelsThatReach_feature"): command += ["-t"]
    if ebudget > -1: command += ["-e", str(ebudget)]

    if(components_by_state): command+=["-z"]
    command += ["-f", str(max_frontier)]

    if verbose: print("Testing agent at ", str(n), " ", str(k), " of ", problem, "from", file, "\n")
    start = time.time()
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    end = time.time()
    if verbose: print("tested in ", end - start, " seconds")

    if proc.returncode == 124:
        results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False}
        #print(results)
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
                results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False,
                           "Exception": True}
                print("Exeption!")
                print(" ".join(command))
                if np.any([("Frontier" in line) for line in err_lines]):
                    print("Frontier did not fit in the buffer.")
                else:
                    for line in lines:
                        print(line)
                    for line in err_lines:
                        print(line)
    expansionMsg = ""
    if(ebudget>-1 and results["expanded transitions"]>ebudget):
        expansionMsg = ", EXPANSION BUDGET EXCEEDED"
    if verbose: print("Total expanded transitions: ",results["expanded transitions"], expansionMsg)
    results["algorithm"] = "new"
    results["heuristic"] = path
    results["problem"] = problem
    results["n"] = n
    results["k"] = k
    results["transitions/ms"] = results["expanded transitions"] / results["synthesis time(ms)"]
    if ebudget > -1 and results["expanded transitions"] > ebudget:
        results["expansion_budget_exceeded"] = "true"
    else:
        results["expansion_budget_exceeded"] = "false"
    return results, debug


def test_monolithic(problem, n, k):
    """ Testing the monolithic approach for a given instance (Huang and Kumar, 2008) """
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
    """ Testing Random for a given instance many times """
    df = []
    start = time.time()
    for i in range(times):
        print("Testing random with", problem, n, k, i, "- Time: ", time.time() - start)
        r = test_agent("mock", problem, n, k, timeout="10m")[0]
        r["idx"] = i
        df.append(r)
    df = pd.DataFrame(df)
    df.to_csv(results_path(problem, n, k, file))


def test_agents(problem, n, k, problem2, n2, k2, file, freq=1):
    df = []

    dir = results_path(problem, n, k, file)
    files = [f for f in os.listdir(dir) if f.endswith(".onnx")]
    for i in range(0, len(files), freq):
        print("Testing", i, "with", problem2, n2, k2)
        path = dir + "/" + str(i) + ".onnx"
        result, debug = test_agent(path, problem2, n2, k2, timeout="10m")

        if result == "timeout":
            result = {"problem": problem2, "n": n2, "k": k2}
        result.update(get_agent_info(path))
        result["idx"] = i
        df.append(result)

    df = pd.DataFrame(df)
    df.to_csv(
        "experiments/results/" + filename([problem, n, k]) + "/" + file + "/" + filename([problem2, n2, k2]) + ".csv")



def test_ra_all_instances(problem, up_to, timeout="10m", name="all_ra", func=test_ra, fast_stop=True, ebudget=-1,
                          solved_crit=budget_and_time):
    """ same parameters as test_agent_all_instances but without the agent path specification (file) """
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            fast_stop_req = (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1])
            full_test_req = (n == 0 or solved[n - 1][k]) or (k == 0 or solved[n][k - 1])
            if k <= 1 and (n > 0 and not solved[n - 1][k]) and (n > 1 and not solved[n - 2][k]):
                full_test_req = False
            if (fast_stop and fast_stop_req) or (not fast_stop and full_test_req):
                print("Testing ra with", problem, n, k)
                df.append(func(problem, n + 1, k + 1, timeout=timeout, ebudget=ebudget)[0])
                if solved_crit(df[-1]):
                    solved[n][k] = True
    print("Solved", np.sum(solved), "instances")
    df = pd.DataFrame(df)
    file = filename([name, up_to, ebudget, timeout]) + ".csv"
    df.to_csv(results_path(problem, 2, 2, file))


def test_agent_all_instances(problem, file, up_to, timeout="10m", name="all", selection=best_generalization_agent_ebudget, max_frontier=1000000,
                             fast_stop=True, ebudget=-1, solved_crit=budget_and_time, total=None, used_testing_timeout = None, used_testing_ebudget = None) :
    """ Step (S3): Testing the selected agent with all instances of a problem """
    """ + *name*: a prefix for the resulting csv name.
        + *selection*: specifies the criterion for selecting the best agent.
        + *fast_stop*: if True, stops evaluation when one previous adjacent context was not solved. if False, it stops evaluation when two of the previous adjacent context were not solved.
        + *solved_crit*: the criterion used to evaluate if a context was solved or not. 
    """
    idx_agent = selection(problem, file,up_to, used_testing_timeout,total,used_testing_ebudget)
    if(idx_agent==None):
        print("No agent solved any context")
        return
    print("Testing all", problem, "with agent: ", idx_agent)
    path = agent_path(problem, file, idx_agent)
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            fast_stop_req = (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1])
            full_test_req = (n == 0 or solved[n - 1][k]) or (k == 0 or solved[n][k - 1])
            if k <= 1 and (n > 0 and not solved[n - 1][k]) and (n > 1 and not solved[n - 2][k]):
                full_test_req = False

            if (fast_stop and fast_stop_req) or (not fast_stop and full_test_req):
                print("Testing agent with", problem, n + 1, k + 1)
                df.append(
                    test_agent(path, problem, n + 1, k + 1, max_frontier=max_frontier, timeout=timeout, ebudget=ebudget,
                               file=file)[0])
                if solved_crit(df[-1]):
                    solved[n][k] = True
    print("Solved", np.sum(solved), "instances")
    df = pd.DataFrame(df)
    df.to_csv(results_path(problem, 2, 2, file) + "/" + name + "_" + problem + "_" + str(up_to) + "_" + str(
        ebudget) + "_TO:" + str(timeout) + ".csv")


def test_random_all_instances(problem, up_to, timeout="10m", name="all_random", ebudget=-1, solved_crit=budget_and_time,
                              file='random'):
    """ same parameters as test_agent_all_instances but without the agent path specification (file) """
    df = []
    solved = [[False for _ in range(up_to)] for _ in range(up_to)]
    for n in range(up_to):
        for k in range(up_to):
            if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                print("Testing random with", problem, n + 1, k + 1)
                df.append(test_agent("mock", problem, n + 1, k + 1, timeout=timeout, ebudget=ebudget, file=file)[0])
                if solved_crit(df[-1]):
                    solved[n][k] = True
    print("Solved", np.sum(solved), "instances")
    df = pd.DataFrame(df)
    df.to_csv(results_path(problem, 2, 2) + "/random" + "/" + name + str(ebudget) + ".csv")


def test_training_agents_generalization(problem, file, up_to, timeout, total=100, max_frontier=1000000,
                                        solved_crit=budget_and_time, ebudget = -1, verbose=True, agentPath = None):
    """ Step (S2): Testing a uniform sample of the trained agents with a reduced budget. """
    """
        + *problem:* a string indicating the name of the problem to test the agent at.
        + *file:* indicates the name of the directory where the agents are to be stored at the ```experiments/results/<used_training_context>``` directory.
        + *up_to* indicates the maximum value used for the $n,k$ possible combinations  of the specified problem to test the specified agent at.
        + *timeout:* a string indicating the time budget allowed for the agent to solve one single context
        + *total:* the size of the used subset.
        + *max_frontier* the budget of possible actions allowed in a single expansion step.
        + *solved_crit* a function that decides whether or not continue testing when the previous adjacent context were not solved (by default, when the adjacent contexts exceeded the expansion and/or time budget).
        + *ebudget*: an integer that specifies the expansions budget. -1 indicates no limit.
    """
    df = []
    start = time.time()
    #breakpoint()
    agents_saved = sorted([int(f[:-5]) for f in os.listdir(agentPath) if "onnx" in f])
    np.random.seed(0)
    tested_agents = sorted(np.random.choice(agents_saved, min(total, len(agents_saved)), replace=False))
    for i in tested_agents:

        solved = [[False for _ in range(up_to)] for _ in range(up_to)]
        if verbose: print("Testing agent", i, "with 5s timeout. Time:", time.time() - start)
        for n in range(up_to):
            for k in range(up_to):
                if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                    df.append(test_agent(agentPath, problem, n + 1, k + 1, max_frontier=max_frontier, timeout=timeout, ebudget=ebudget, file = file, verbose = False)[0])
                    df[-1]["idx"] = i
                    if solved_crit(df[-1]):
                        solved[n][k] = True
        if verbose: print("Solved:", np.sum(solved))

    df = pd.DataFrame(df)
    df.to_csv(agentPath + "/generalization_all" + joinAsStrings([up_to, timeout,total,ebudget])+ ".csv")


def test_training_agents_generalization_2(problem, file, extrapolation_space, timeout, total=100, max_frontier=1000000,
                                        solved_crit=budget_and_time, ebudget = -1, verbose=True, agentPath = None,
                                          path_to_analysis = "./generalization_all.csv", components_by_state = False):
    """ Step (S2): Testing a uniform sample of the trained agents with a reduced budget. """
    """
        + *problem:* a string indicating the name of the problem to test the agent at.
        + *file:* indicates the name of the directory where the agents are to be stored at the ```experiments/results/<used_training_context>``` directory.
        + *up_to* indicates the maximum value used for the $n,k$ possible combinations  of the specified problem to test the specified agent at.
        + *timeout:* a string indicating the time budget allowed for the agent to solve one single context
        + *total:* the size of the used subset.
        + *max_frontier* the budget of possible actions allowed in a single expansion step.
        + *solved_crit* a function that decides whether or not continue testing when the previous adjacent context were not solved (by default, when the adjacent contexts exceeded the expansion and/or time budget).
        + *ebudget*: an integer that specifies the expansions budget. -1 indicates no limit.
    """
    df = []
    start = time.time()
    #breakpoint()
    agents_saved = sorted([int(f[:-5]) for f in os.listdir(agentPath) if "onnx" in f])
    np.random.seed(0)

    tested_agents = sorted(np.random.choice(agents_saved, min(total, len(agents_saved)), replace=False))

    extrapolation_space.sort()
    up_to = max(extrapolation_space[-1])
    for i in tested_agents:
        solved = [[False for _ in range(up_to)] for _ in range(up_to)]
        if verbose: print("Testing agent", i, "with 5s timeout. Time:", time.time() - start)
        for n in range(up_to):
            for k in range(up_to):
                if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                    #FIXME por alguna razon el if siempre es falso y (n,k)=(0,0)
                    print((n,k))
                    if((n + 1,k + 1 ) in extrapolation_space):
                        df.append(test_agent(agentPath, problem, n + 1, k + 1, max_frontier=max_frontier,timeout=timeout, ebudget=ebudget, file = file, verbose = False,components_by_state = components_by_state)[0])
                        print("tested ", n, k)
                    else: continue
                    df[-1]["idx"] = i
                    if solved_crit(df[-1]):
                        solved[n][k] = True
        if verbose: print("Solved:", np.sum(solved))

    df = pd.DataFrame(df)
    df.to_csv(path_to_analysis)


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


if __name__ == "__main__":
    problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
    for p in problems:
        if not os.path.isdir("results/" + p):
            os.makedirs("results/" + p)
    file = sys.argv[1]
    for problem in problems:
        test_ra_all_instances(problem=problem,up_to=15,timeout="10m", ebudget=-1,solved_crit=budget_and_time)
        test_ra_all_instances(problem=problem,up_to=15,timeout="3h", ebudget=15000,solved_crit=budget_and_time)
        test_random_all_instances(problem=problem,up_to=15,timeout="10m",ebudget=-1,solved_crit=budget_and_time)
        test_random_all_instances(problem=problem,up_to=15, timeout="3h", ebudget=15000, solved_crit=budget_and_time)
