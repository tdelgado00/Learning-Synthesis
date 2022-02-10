import pickle
import time
from copy import deepcopy

import numpy as np

import os


class AgentSaver:

    def __init__(self, dir):
        self.agents = []
        self.dir = dir
        self.idx = 0

    def save(self, agent, training_time, steps):
        os.makedirs(self.dir, exist_ok=True)
        with open(self.dir+"/"+str(self.idx)+".pkl", "wb") as f:
            start = time.time()
            pickle.dump((agent, training_time, steps), f)
            self.idx += 1

            print("Agent saved. Training time: ", training_time)
            print("Saving time:", time.time() - start)


class Agent:

    def __init__(self, model, dir=None):
        self.model = model
        self.has_learned_something = False

        self.agent_saver = AgentSaver(dir) if dir is not None else None

    def test(self, env, timeout=30*60):
        start_time = time.time()

        obs = env.reset()
        done = False
        info = None

        while not done and time.time() - start_time < timeout:
            action = self.get_action(obs, 0)
            obs, reward, done, info = env.step(action)

        return info if time.time() - start_time < timeout else "timeout"

    def train(self, env, seconds, copy_freq=200000, epsilon=0.1):
        training_start = time.time()

        steps = 0
        obs = env.reset()
        while time.time() - training_start < seconds:
            a = self.get_action(obs, epsilon)
            a_features = obs[a]
            obs, reward, done, _ = env.step(a)

            if done:
                self.update(a_features, reward)
                obs = env.reset()
            else:
                self.update(a_features, reward + np.max(self.eval(obs)))
            steps += 1

            if steps % copy_freq == 0 and self.agent_saver is not None:
                training_time = time.time() - training_start
                self.agent_saver.save(self, training_time, steps)

    def get_action(self, actionFeatures, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(len(actionFeatures))
        else:
            return np.argmax(self.eval(actionFeatures))

    def eval(self, actionFeatures):
        if not self.has_learned_something:
            return np.random.rand(len(actionFeatures))
        values = self.model.predict(actionFeatures)
        return values

    def update(self, features, value):
        self.model.partial_fit([features], [value])
        self.has_learned_something = True
