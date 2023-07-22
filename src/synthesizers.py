import random
import warnings

import onnx
import onnxruntime

from environment import CompositionGraph
from util import *
import subprocess
import time
from util import parse_java_debug
import sys

class Synthesizer:
    """ Wrapper for the execution of a command that evaluates a synthesis technique in a given instance of a problem
    """

    def __init__(self, problem):
        self.problem = problem

    def base_command(self):
        raise NotImplementedError()

    def read_results(self, lines, err_lines, command_run):
        raise NotImplementedError()

    def heuristic_name(self):
        raise NotImplementedError()

    def algorithm_name(self):
        raise NotImplementedError()


    def test(self, n, k, ebudget=-1, timeout="10h", verbose=False):
        command = self.base_command()
        command = ["timeout", timeout] + command + ["-i", fsp_path(self.problem, n, k)]

        if ebudget > -1:
            command += ["-e", str(ebudget)]

        if verbose:
            print("Testing", self.heuristic_name(), "on", self.problem, n, k)

        start = time.time()
        proc = subprocess.run(command,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        end = time.time()

        if verbose:
            print("Finished in", end - start, "seconds")

        if proc.returncode == 124:
            results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False}

        else:
            lines = proc.stdout.split("\n")[2:]
            err_lines = proc.stderr.split("\n")
            results = self.read_results(lines, err_lines, command)

        expansion_msg = ""
        if -1 < ebudget < results["expanded transitions"]:
            expansion_msg = "- EXPANSION BUDGET EXCEEDED"

        if verbose:
            print("Total expanded transitions:", results["expanded transitions"], expansion_msg)

        results["algorithm"] = self.algorithm_name()
        results["heuristic"] = self.heuristic_name()
        results["problem"] = self.problem
        results["n"] = n
        results["k"] = k
        results["transitions/ms"] = results["expanded transitions"] / results["synthesis time(ms)"]

        if -1 < ebudget < results["expanded transitions"]:
            results["expansion_budget_exceeded"] = "true"
        else:
            results["expansion_budget_exceeded"] = "false"

        return results


class OnTheFlyFeatureBased(Synthesizer):

    def __init__(self, problem, max_frontier=1000000, debug=False):
        super().__init__(problem)
        self.debug = debug
        self.max_frontier = max_frontier
        self.debug_output = None

    def heuristic_name(self):
        raise NotImplementedError()

    def algorithm_name(self):
        return "new"  # Kept for backwards compatibility

    def features_arg(self):
        raise NotImplementedError()

    def base_command(self):
        command = ["java", "-Xmx8g", "-XX:MaxDirectMemorySize=512m", "-classpath", "mtsa.jar",
                   "MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking.FeatureBasedExplorationHeuristic"]

        command += ["-m", self.heuristic_name()]

        if self.debug:
            command += ["-d"]

        command += self.features_arg()

        command += ["-f", str(self.max_frontier)]

        return command

    def read_results(self, lines, err_lines, command_run):

        if np.any(["OutOfMem" in line for line in err_lines]):
            self.debug_output = None
            results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": True}

        else:
            try:
                i = list(map((lambda l: "ExpandedStates" in l), lines)).index(True)
                j = list(map((lambda l: "DirectedController" in l), lines)).index(True)
                if self.debug:
                    self.debug_output = parse_java_debug(lines[:j])
                    results = read_results(lines[i:])
                else:
                    self.debug_output = None
                    results = read_results(lines[i:])
                results["OutOfMem"] = False

            except BaseException as err:
                results = {"expanded transitions": np.nan, "synthesis time(ms)": np.nan, "OutOfMem": False,
                           "Exception": True}

                print("Exception!", " ".join(command_run))

                if np.any([("Frontier" in line) for line in err_lines]):
                    print("Frontier did not fit in the buffer.")
                else:
                    for line in lines:
                        print(line)
                    for line in err_lines:
                        print(line)

        return results


class OnTheFlyRL(OnTheFlyFeatureBased):

    def __init__(self, path, features_path, problem, max_frontier=1000000, debug=False):
        super().__init__(problem, max_frontier=max_frontier, debug=debug)
        self.path = path
        self.features_path = features_path

    def heuristic_name(self):
        return self.path

    def features_arg(self):
        arg = ["-c", self.features_path]

        with open(self.features_path, "r") as f:
            using_labels = "labels 1" in [x[:-1] for x in f]
            if using_labels:
                arg += ["-l", "./labels/" + self.problem + ".txt"]
        return arg

class OnTheFlyRLFromPython(OnTheFlyRL):
    def __init__(self, path, features_path, problem, max_frontier=1000000, debug=False):
        super().__init__(path, features_path, problem, max_frontier=1000000, debug=False)
        self.model = onnxruntime.InferenceSession(self.path)
    def test_from_python(self, n, k):
        warnings.warn("Still testing")
        composition = CompositionGraph(self.problem,n,k)
        composition.start_composition()

        while (len(composition.getFrontier())>0 and not composition.finished()):
            frontier = composition.getFrontier()
            composition.expand(random.randint(0,len(frontier)-1))

        breakpoint()
        print("finished!", len(composition.nodes), " ",  len(composition.edges))

        #outputs = ort_session.run(None,{"X": np.random.randn(batch_size, 34).astype(np.float32)},)



class OnTheFlyRandom(OnTheFlyFeatureBased):
    def heuristic_name(self):
        return "random"

    def features_arg(self):
        return []


class OnTheFlyRA(Synthesizer):
    def __init__(self, problem):
        super().__init__(problem)
        self.debug_output = None

    def heuristic_name(self):
        return "random"

    def algorithm_name(self):
        return "OpenSet RA"  # Kept for backwards compatibility

    def base_command(self):
        command = ["java", "-Xmx8g", "-classpath", "mtsa.jar",
                   "ltsa.ui.LTSABatch", "-c", "DirectedController", "-r"]

        return command

    def read_results(self, lines, err_lines, command_run):
        return read_results(lines)


class Monolithic(Synthesizer):
    """ A monolithic approach (Huang and Kumar, 2008) """

    def __init__(self, problem):
        super().__init__(problem)
        self.debug_output = None

    def heuristic_name(self):
        return None

    def algorithm_name(self):
        return "mono"

    def base_command(self):
        command = ["java", "-cp", "mtsaOld.jar", "ltsa.ui.LTSABatch", "-c", "MonolithicController"]

        return command

    def read_results(self, lines, err_lines, command_run):
        return read_results(lines)

if __name__=="__main__":
    at_otf_rl = OnTheFlyRLFromPython("experiments/results/check_correct_solutions/AT/8.onnx",features_path="experiments/results/check_correct_solutions/AT/features.txt",problem="AT")
    at_otf_rl.test_from_python(3,3)
