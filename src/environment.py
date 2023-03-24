import jpype
import jpype.imports
from util import *
import copy
from plots import read_monolithic

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['mtsa.jar'])
from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DCSForPython, FeatureBasedExplorationHeuristic


class DCSSolverEnv:
    def __init__(self, problem, n, k, features, normalize_reward=False):
        self.problem = problem
        self.n = n
        self.k = k
        self.features = features
        self.problemFilename = filename([problem, n, k])
        self.normalize_reward = normalize_reward
        self.problem_size = read_monolithic()[("expanded transitions", problem)][k][n]
        self.detached_initial_componentwise_info = None
        if features["components_by_state"]:
            self.detached_initial_componentwise_info = FeatureBasedExplorationHeuristic.compileFSP("fsp/"+problem+"/"+"-".join([problem, str(n), str(k)])+".fsp").getFirst()
        self.javaEnv = DCSForPython("labels/" + problem + ".txt" if features["labels"] else "mock", 10000,
                                    features["ra feature"],
                                    features["context features"],
                                    features["state labels"],
                                    features["je feature"],
                                    features["nk feature"],
                                    features["prop feature"],
                                    features["visits feature"],
                                    features["labelsThatReach_feature"],
                                    features["only boolean"], self.detached_initial_componentwise_info
                                    )
        self.nfeatures = self.javaEnv.getNumberOfFeatures()
        self.info = dict(features)
        self.info.update({
            "nfeatures": self.nfeatures,
            "n": self.n,
            "k": self.k,
            "problem": self.problem,
            "expansion_budget_exceeded": "false"
        })

    def get_actions(self):
        nactions = self.javaEnv.frontierSize()
        actions = np.asarray(self.javaEnv.input_buffer)
        r = actions[:nactions * self.nfeatures].reshape((nactions, self.nfeatures)).copy()
        return r

    def step(self, action):
        self.javaEnv.expandAction(action)
        if not self.javaEnv.isFinished():
            return self.get_actions(), self.reward(), False, {}
        else:
            return None, self.reward(), True, self.get_results()

    def reward(self):
        return -1 if not self.normalize_reward else -1 / self.problem_size

    def reset(self):
        self.javaEnv.startSynthesis("fsp/"+self.problem+"/"+"-".join([self.problem, str(self.n), str(self.k)])+".fsp")
        return self.get_actions()

    def close(self):
        pass

    def get_results(self):
        return {
            "synthesis time(ms)": float(self.javaEnv.getSynthesisTime()),
            "expanded transitions": int(self.javaEnv.getExpandedTransitions()),
            "expanded states": int(self.javaEnv.getExpandedStates()),
        }
def generateEnvironments(instances, features, ebudget=-1):
    env = {}
    for instance in instances:
        problem, n, k = instance
        env[instance] = DCSSolverEnv(problem, n, k, features)
    return env

if __name__ == "__main__":
    env = DCSSolverEnv("AT", 3, 3)
    obs = env.reset()
    done = False
    info = None
    while not done:
        action = np.random.randint(len(obs))
        obs, reward, done, info = env.step(action)

    print(info)
