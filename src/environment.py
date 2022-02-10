from time import sleep

import numpy as np
import jpype
import jpype.imports
from util import *
import gym
from gym import spaces

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['mtsa.jar'])
from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DCSForPythonFF
from MTSTools.ac.ic.doc.mtstools.model.operations.DCS.nonblocking import DCSForPythonPF


class DCSSolverEnv(gym.Env):
    def __init__(self, problem, n, k, nactions=None, max_actions=3000):
        super(DCSSolverEnv, self).__init__()
        self.problem = problem
        self.n = n
        self.k = k
        self.problemFilename = filename([problem, n, k])

        if nactions is None:
            self.max_actions = max_actions
            self.javaEnv = DCSForPythonFF()
            self.nfeatures = self.javaEnv.getNumberOfFeatures()
            self.featuresBuffer = jpype.nio.convertToDirectBuffer(
                bytearray(self.nfeatures * self.max_actions * 4)).asFloatBuffer()
            self.javaEnv.setFeaturesBuffer(self.featuresBuffer)
            self.get_actions = self.get_actions_full_frontier
        else:
            self.nactions = nactions
            self.binaryFeaturesBuffer = jpype.nio.convertToDirectBuffer(bytearray(6 * self.nactions))
            self.boxFeaturesBuffer = jpype.nio.convertToDirectBuffer(bytearray(4 * 4 * self.nactions)).asFloatBuffer()

            self.javaEnv = DCSForPythonPF(self.nactions, self.boxFeaturesBuffer, self.binaryFeaturesBuffer)

            self.action_space = spaces.Discrete(self.nactions)
            self.observation_space = spaces.Dict({
                    "binary": spaces.MultiBinary(6 * self.nactions),
                    "real": spaces.Box(low=0, high=1, shape=(4 * self.nactions,), dtype=np.float32)
                })
            self.get_actions = self.get_actions_partial_frontier


    def get_actions_full_frontier(self):
        nactions = self.javaEnv.frontierSize()
        actions = np.asarray(self.featuresBuffer, dtype=np.float32)
        return actions[:nactions * self.nfeatures].reshape((nactions, self.nfeatures))

    def get_actions_partial_frontier(self):
        return {
            "binary": np.asarray(self.binaryFeaturesBuffer, dtype=np.bool),
            "real": np.asarray(self.boxFeaturesBuffer, dtype=np.float32)
        }

    def step(self, action):
        self.javaEnv.expandAction(action)
        if not self.javaEnv.isFinished():
            return self.get_actions(), self.reward(False), False, {}
        else:
            return None, self.reward(True), True, self.get_results()

    def reward(self, done):
        return 0 if done else -1

    def reset(self):
        self.javaEnv.startSynthesis(self.problem, self.n, self.k, "")
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