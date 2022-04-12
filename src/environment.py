import numpy as np
import jpype
import jpype.imports
from util import *

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['mtsa.jar'])
from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DCSForPython


class DCSSolverEnv:
    def __init__(self, problem, n, k, ra_feature):
        super(DCSSolverEnv, self).__init__()
        self.problem = problem
        self.n = n
        self.k = k
        self.problemFilename = filename([problem, n, k])

        self.javaEnv = DCSForPython("", "mock", ra_feature)
        self.nfeatures = self.javaEnv.getNumberOfFeatures()
        self.featuresBuffer = jpype.nio.convertToDirectBuffer(
            bytearray(self.nfeatures * 100000 * 4)).asFloatBuffer()
        self.javaEnv.setFeaturesBuffer(self.featuresBuffer)

    def get_actions(self):
        nactions = self.javaEnv.frontierSize()
        actions = np.asarray(self.featuresBuffer, dtype=np.float32)
        return actions[:nactions * self.nfeatures].reshape((nactions, self.nfeatures)).copy()

    def step(self, action):
        self.javaEnv.expandAction(action)
        if not self.javaEnv.isFinished():
            return self.get_actions(), self.reward(False), False, {}
        else:
            return None, self.reward(True), True, self.get_results()

    def reward(self, done):
        return 0 if done else -1

    def reset(self):
        self.javaEnv.startSynthesis(self.problem, self.n, self.k)
        return self.get_actions()

    def close(self):
        pass

    def get_results(self):
        return {
                "synthesis time(ms)": float(self.javaEnv.getSynthesisTime()),
                "expanded transitions": int(self.javaEnv.getExpandedTransitions()),
                "expanded states": int(self.javaEnv.getExpandedStates()),
                "n": self.n,
                "k": self.k,
                "problem": self.problem,
                "nfeatures": self.javaEnv.getNumberOfFeatures()
        }

if __name__ == "__main__":
    env = DCSSolverEnv("AT", 3, 3)
    obs = env.reset()
    done = False
    info = None
    while not done:
        action = np.random.randint(len(obs))
        obs, reward, done, info = env.step(action)

    print(info)