import warnings
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

FSP_PATH = "./fsp"
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

        print("Warning: max frontier size modified to 1000000, remember baseline uses 10000")

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
        breakpoint()
        print("HOW DO YOU ITERATE OVER THE FRONTIER INSIDE THE NX.GRAPH TO MATCH THE CORRESPONDING INDEX?")
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
        self.exploration_graph.add_edge(child_compostate[0], child_compostate[2], controllability=int(child_compostate[4]), label=getTransitionType(child_compostate[1]))

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

class CompositionGraph(nx.DiGraph):
    def __init__(self, problem, n, k):
        super().__init__()
        self._problem, self._n, self._k = problem, n , k
        self._initial_state = None
        self._state_machines  = [] # : list[nx.DiGraph]
        self._frontier = []
        self._started, self._completed = False, False
        self._javaEnv = None
        self._alphabet = []
        self._no_indices_alphabet = []
        self._number_of_goals = 0
        self._expansion_order = []

    def start_composition(self, mtsa_version_path = 'mtsa.jar'):
        assert(self._initial_state is None)
        print("Warning: underlying Java code runs unused feature computations and buffers")
        if not jpype.isJVMStarted(): jpype.startJVM(classpath=[mtsa_version_path])
        from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DirectedControllerSynthesisNonBlocking, FeatureBasedExplorationHeuristic, DCSForPython
        from ltsa.dispatcher import TransitionSystemDispatcher
        self._started = True
        c = FeatureBasedExplorationHeuristic.compileFSP(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        ltss_init = c.getFirst()
        self._state_machines = [m.name for m in ltss_init.machines] #TODO: turn it into a dictionary that goes from the state machine name into its respective digraph
        self._javaEnv = DCSForPython(None, None, 10000, ltss_init)
        self._javaEnv.startSynthesis(f"{FSP_PATH}/{self._problem}/{self._problem}-{self._n}-{self._k}.fsp")
        assert(self._javaEnv is not None)
        self._initial_state = self._javaEnv.dcs.initial
        self.add_node(self._initial_state)
        self._alphabet = [e for e in self._javaEnv.dcs.alphabet.actions]
        self._alphabet.sort()



    def expand(self, idx):
        assert(not self._javaEnv.isFinished()), "Invalid expansion, composition is already solved"
        assert (idx<len(self.getFrontier()) and idx>=0), "Invalid index"
        self._javaEnv.expandAction(idx)
        new_state_action = self.getLastExpanded()
        controllability, label = self.getLastExpanded().action.isControllable(), self.getLastExpanded().action.toString()
        self.add_node(self.last_expansion_child_state())
        self.add_edge(new_state_action.state, self.last_expansion_child_state(), controllability=controllability, label=label, action_with_features = new_state_action)
        self._expansion_order.append(self.getLastExpanded())
    def last_expansion_child_state(self):
        return self._javaEnv.heuristic.lastExpandedTo
    def last_expansion_source_state(self):
        return self._javaEnv.heuristic.lastExpandedFrom
    def getFrontier(self): return self._javaEnv.heuristic.explorationFrontier
    def getLastExpanded(self): return self._javaEnv.heuristic.lastExpandedStateAction

    def _check_no_repeated_states(self):
        raise NotImplementedError

    def explored(self, transition):
        """
        Whether a transition from s or s ′ has
            already been explored.

        """
        raise NotImplementedError
    def last_expanded(self, transition):
        """Whether s is the last expanded state in h
            (outgoing or incoming)."""
        raise NotImplementedError

    def finished(self):
        return self._javaEnv.isFinished()
class CompositionAnalyzer:
    """class used to get Composition information, usable as hand-crafted features"""

    def __init__(self, composition : CompositionGraph):
        self.composition = composition
        assert (self.composition._started)

        self._no_indices_alphabet = list(set([self.remove_indices(str(e)) for e in composition._alphabet]))
        self._no_indices_alphabet.sort()
        self._fast_no_indices_alphabet_dict = dict()
        for i in range(len(self._no_indices_alphabet)): self._fast_no_indices_alphabet_dict[self._no_indices_alphabet[i]]=i
        self._fast_no_indices_alphabet_dict = bidict(self._fast_no_indices_alphabet_dict)
        self._feature_methods = [self.event_label_feature,self.state_label_feature,self.controllable
                                ,self.marked_state, self.current_phase,self.child_node_state,
                                 self.uncontrollable_neighborhood, self.explored_state_child, self.isLastExpanded]


    def test_features_on_transition(self, transition):
        [compute_feature(transition) for compute_feature in self._feature_methods]
    def event_label_feature(self, transition):
        """
        Determines the label of ℓ in A E p .
        """
        feature_vec_slice = [0 for _ in self._no_indices_alphabet]
        self._set_transition_type_bit(feature_vec_slice, transition.action)
        #print(no_idx_label, feature_vec_slice)
        return feature_vec_slice

    def _set_transition_type_bit(self, feature_vec_slice, transition):
        no_idx_label = self.remove_indices(transition.toString())
        feature_vec_slice_pos = self._fast_no_indices_alphabet_dict[no_idx_label]
        feature_vec_slice[feature_vec_slice_pos] = 1

    def state_label_feature(self, transition):
        """
        Determines the labels of the explored
            transitions that arrive at s.
        """
        feature_vec_slice = [0 for _ in self._no_indices_alphabet]
        arriving_to_s = transition.state.getParents()
        for trans in arriving_to_s: self._set_transition_type_bit(feature_vec_slice,trans.getFirst())
        return feature_vec_slice
    def controllable(self, transition):
        return [int(transition.action.isControllable())]
    def marked_state(self, transition):
        """Whether s and s ′ ∈ M E p ."""
        return [int(transition.childMarked)]

    def current_phase(self, transition):
        return [int(self.composition._javaEnv.dcs.heuristic.goals_found > 0),
                int(self.composition._javaEnv.dcs.heuristic.marked_states_found > 0),
                int(self.composition._javaEnv.dcs.heuristic.closed_potentially_winning_loops > 0)]



    def child_node_state(self, transition):
        """Whether
        s ′ is winning, losing, none,
        or not yet
        explored."""
        res = [0, 0, 0]
        if(transition.child is not None):
            res = [int(transition.child.status.toString()=="GOAL"),
                   int(transition.child.status.toString()=="ERROR"),
                   int(transition.child.status.toString()=="NONE")]
        return res
    def uncontrollable_neighborhood(self, transition):
        warnings.warn("Chequear que este bien")
        return [int(transition.state.uncontrollableUnexploredTransitions>0),
                int(transition.state.uncontrollableTransitions>0),
                int(transition.child is None or transition.child.uncontrollableUnexploredTransitions > 0),
                int(transition.child is None or transition.child.uncontrollableTransitions > 0)
                ]

    def explored_state_child(self, transition):
        return [int(len(self.composition.out_edges(transition.state))!= transition.state.unexploredTransitions),
                int(transition.child is not None and len(self.composition.out_edges(transition.child))!= transition.state.unexploredTransitions)]

    def isLastExpanded(self, transition):
        warnings.warn("For some reason, sometimes no edge in the entire graph was the las one expanded!")
        return [int(self.composition.getLastExpanded()==transition)]

    def remove_indices(self, transition_label : str):
        res = ""

        for c in transition_label:
            if not c.isdigit(): res += c

        return res
    def compute_features(self, transition):
        res = []
        for feature_method in self._feature_methods:
            res += feature_method(transition)
        return res
