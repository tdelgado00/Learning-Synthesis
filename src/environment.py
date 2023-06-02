from collections import OrderedDict

import jpype
import jpype.imports
from sympy.core.containers import OrderedSet

from util import *
import os
import pickle
import networkx as nx

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['mtsa.jar'])
from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DCSForPython, FeatureBasedExplorationHeuristic


class DCSSolverEnv:
    def __init__(self, problem, n, k, features_path, normalize_reward=False, exploration_graph=False):
        self.problem = problem
        self.n = n
        self.k = k
        self.problemFilename = filename([problem, n, k])
        self.normalize_reward = normalize_reward
        self.problem_size = read_monolithic()[("expanded transitions", problem)][k][n]
        self.detached_initial_componentwise_info = None

        cbs_enabled = False
        labels_enabled = False
        with open(features_path, "r+") as f:
            lines = [line[:-1] for line in f]
            for line in lines:
                if line.startswith("cbs"):
                    if int(line.split(" ")[1]) == 1:
                        cbs_enabled = True
                elif line.startswith("labels"):
                    if int(line.split(" ")[1]) == 1:
                        labels_enabled = True

        if cbs_enabled:
            self.detached_initial_componentwise_info = FeatureBasedExplorationHeuristic.compileFSP(
                "fsp/" + problem + "/" + "-".join([problem, str(n), str(k)]) + ".fsp").getFirst()

        labels_path = "labels/" + problem + ".txt" if labels_enabled else None

        self.javaEnv = DCSForPython(features_path,
                                    labels_path,
                                    10000,
                                    self.detached_initial_componentwise_info
                                    )

        self.nfeatures = self.javaEnv.getNumberOfFeatures()
        self.transition_labels = OrderedSet()

        self.info = {
            "nfeatures": self.nfeatures,
            "n": self.n,
            "k": self.k,
            "problem": self.problem,
            "expansion_budget_exceeded": "false"
        }

        self.exploration_graph = None
        if exploration_graph:
            self.exploration_graph = nx.DiGraph()

    def set_transition_types(self):
        for transition_label in self.javaEnv.all_transition_types():
            self.transition_labels.add(getTransitionType(transition_label))

    def get_actions(self):
        nactions = self.javaEnv.frontierSize()
        actions = np.asarray(self.javaEnv.input_buffer)
        r = actions[:nactions * self.nfeatures].reshape((nactions, self.nfeatures)).copy()
        return r

    def step(self, action):
        if self.exploration_graph is not None:
            self.featured_graph_expansion(action)
        else:
            self.javaEnv.expandAction(action)

        if not self.javaEnv.isFinished():
            return self.get_actions(), self.reward(), False, {}
        else:
            return None, self.reward(), True, self.get_results()

    def featured_graph_expansion(self, action):
        self.javaEnv.expandAction(action)
        child_compostate_java_format = self.javaEnv.lastExpandedStringIdentifiers()
        child_compostate = [str(e) for e in child_compostate_java_format]
        child_features = self.compute_node_features(child_compostate)
        if child_compostate[0] not in self.exploration_graph.nodes():
            # print(child_compostate[0], type(child_compostate[0]))
            self.exploration_graph.add_node(child_compostate[0], features=[1], marked=0)  # TODO: this is hardcoded
        if child_compostate[2] not in self.exploration_graph.nodes():
            # print(child_compostate[2], type(child_compostate[2]))
            self.exploration_graph.add_node(child_compostate[2], features=child_features, marked=int(child_compostate[3]))
        self.exploration_graph.add_edge(child_compostate[0], child_compostate[2], controllability=int(child_compostate[4]))  #, label=child_compostate[1])

    def compute_node_features(self, child_compostate):
        return [1]
        # child_is_unmarked = 1 - int(str(child_compostate[3]))
        # child_features = [child_is_unmarked]
        # return child_features

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


def getTransitionType(full_transition_label):
    i = 0
    res = ""
    while i < len(full_transition_label) and full_transition_label[i] != '.':
        res += (full_transition_label[i])
        i += 1
    return res


def save_random_states(problems, n, k, features):
    """ Saves observations from a random policy for all problems in the benchmark """

    def get_random_states(env, total=20000, sampled=2000):
        idxs = np.random.choice(range(total), sampled)

        states = []
        done = True
        obs = None
        for i in range(total):
            if done:
                obs = env.reset()

            if i in idxs:
                states.append(np.copy(obs))

            action = np.random.randint(len(obs))
            obs, reward, done, info = env.step(action)

        return states

    for problem in problems:
        print("Saving random states for problem", problem)
        states = get_random_states(DCSSolverEnv(problem, n, k, features))

        file = results_path(problem, n, k, "states_b.pkl")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(states, f)
