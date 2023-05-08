import jpype
import jpype.imports
from util import *
from plots import read_monolithic
import networkx as nx
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['mtsa.jar'])
from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DCSForPython, FeatureBasedExplorationHeuristic


class DCSSolverEnv:
    def __init__(self, problem, n, k, args, features_path, normalize_reward=False, exploration_graph=False):
        self.problem = problem
        self.n = n
        self.k = k
        self.problemFilename = filename([problem, n, k])
        self.normalize_reward = normalize_reward
        self.problem_size = read_monolithic()[("expanded transitions", problem)][k][n]
        self.detached_initial_componentwise_info = None

        if args.cbs:
            self.detached_initial_componentwise_info = FeatureBasedExplorationHeuristic.compileFSP(
                "fsp/" + problem + "/" + "-".join([problem, str(n), str(k)]) + ".fsp").getFirst()

        labels_path = "labels/" + problem + ".txt" if args.labels else None

        print(features_path, labels_path)
        self.javaEnv = DCSForPython(features_path,
                                    labels_path,
                                    10000,
                                    self.detached_initial_componentwise_info
                                    )

        self.nfeatures = self.javaEnv.getNumberOfFeatures()
        self.info = {
            "nfeatures": self.nfeatures,
            "n": self.n,
            "k": self.k,
            "problem": self.problem,
            "expansion_budget_exceeded": "false"
        }
        self.exploration_graph = None
        if exploration_graph: self.exploration_graph = nx.DiGraph()

    def get_actions(self):
        nactions = self.javaEnv.frontierSize()
        actions = np.asarray(self.javaEnv.input_buffer)
        r = actions[:nactions * self.nfeatures].reshape((nactions, self.nfeatures)).copy()
        return r

    def step(self, action):
        if self.exploration_graph is not None:
            child_compostate = self.javaEnv.expandAction(action)
            child_is_marked = int(child_compostate[3])
            child_features = [child_is_marked]

            if child_compostate[0] not in self.exploration_graph.nodes(): self.exploration_graph.add_node(child_compostate[0], features = [0])
            if child_compostate[2] not in self.exploration_graph.nodes():self.exploration_graph.add_node(child_compostate[2], features = child_features)
            self.exploration_graph.add_edge(child_compostate[0], child_compostate[2], label=child_compostate[1])
        else:
            self.javaEnv.expandAction(action)

        if not self.javaEnv.isFinished():
            return self.get_actions(), self.reward(), False, {}
        else:
            return None, self.reward(), True, self.get_results()

    def reward(self):
        return -1 if not self.normalize_reward else -1 / self.problem_size

    def reset(self):
        self.javaEnv.startSynthesis(
            "fsp/" + self.problem + "/" + "-".join([self.problem, str(self.n), str(self.k)]) + ".fsp")
        return self.get_actions()

    def close(self):
        pass

    def get_results(self):
        return {
            "synthesis time(ms)": float(self.javaEnv.getSynthesisTime()),
            "expanded transitions": int(self.javaEnv.getExpandedTransitions()),
            "expanded states": int(self.javaEnv.getExpandedStates())
        }
