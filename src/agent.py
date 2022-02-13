import json
import time

import numpy as np

import os

import onnx
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from sklearn.neural_network import MLPRegressor


class Agent:
    def __init__(self, eta=1e-5, nnsize=20, epsilon=0.1, dir=None):

        self.model = MLPRegressor(hidden_layer_sizes=(nnsize,),
                                  solver="sgd",
                                  learning_rate="constant",
                                  learning_rate_init=eta)
        self.has_learned_something = False

        self.eta = eta
        self.nnsize = nnsize
        self.epsilon = epsilon

        self.dir = dir
        self.save_idx = 0

    def test(self, env, timeout=30 * 60):
        start_time = time.time()

        obs = env.reset()
        done = False
        info = None

        while not done and time.time() - start_time < timeout:
            action = self.get_action(obs, 0)
            obs, reward, done, info = env.step(action)

        return info if time.time() - start_time < timeout else "timeout"

    def train(self, env, seconds, copy_freq=200000):
        training_start = time.time()
        saving_time = 0

        steps = 0

        obs = env.reset()
        while time.time() - training_start - saving_time < seconds:
            a = self.get_action(obs, self.epsilon)
            a_features = obs[a]
            obs, reward, done, _ = env.step(a)

            if done:
                self.update(a_features, reward)
                obs = env.reset()
            else:
                self.update(a_features, reward + np.max(self.eval(obs)))
            steps += 1

            if steps % copy_freq == 0 and self.dir is not None:
                self.save(time.time() - training_start, steps)

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

    def save(self, training_time, steps):
        os.makedirs(self.dir, exist_ok=True)
        X_test = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]).astype(np.float32)
        onx = to_onnx(self.model, X_test)
        onnx.save(onx, self.dir + "/" + str(self.save_idx) + ".onnx")

        with open(self.dir + "/" + str(self.save_idx) + ".json", "w") as f:
            info = {
                "training time": training_time,
                "training steps": steps,
                "eta": self.eta,
                "nnsize": self.nnsize,
                "epsilon": self.epsilon,
            }
            json.dump(info, f)

        print("Agent", self.save_idx, "saved. Training time:", training_time)
        self.save_idx += 1


def test_onnx(model, env, timeout=30*60):
    start_time = time.time()
    sess = InferenceSession(model.SerializeToString())

    obs = env.reset()
    done = False
    info = None

    while not done and time.time() - start_time < timeout:
        action = np.argmax(sess.run(None, {'X': obs}))
        obs, reward, done, info = env.step(action)

    return info if time.time() - start_time < timeout else {
        "problem": env.problem,
        "n": env.n,
        "k": env.k,
        "synthesis time(ms)": np.nan,
        "expanded transitions": np.nan
    }