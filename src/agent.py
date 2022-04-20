import json
import time

import numpy as np

import os

import onnx
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from sklearn.neural_network import MLPRegressor
from modelEvaluation import get_random_experience
from replayBuffer import ReplayBuffer


class Agent:
    def __init__(self, eta=1e-5, nnsize=20, epsilon=0.1, dir=None, fixed_q_target=False, reset_target_freq=10000,
                 experience_replay=False, buffer_size=10000, batch_size=32, verbose=False):

        self.model = MLPRegressor(hidden_layer_sizes=(nnsize,),
                                  solver="sgd",
                                  learning_rate="constant",
                                  learning_rate_init=eta)

        self.fixed_q_target = fixed_q_target
        self.reset_target_freq = reset_target_freq
        self.target = None

        self.has_learned_something = False

        self.experience_replay = experience_replay
        self.buffer_size = buffer_size
        self.buffer = None
        self.batch_size = batch_size

        self.eta = eta
        self.nnsize = nnsize
        self.epsilon = epsilon

        self.dir = dir
        self.save_idx = 0

        self.verbose = verbose

    def test(self, env, timeout=30 * 60):
        start_time = time.time()

        obs = env.reset()
        done = False
        info = None

        while not done and time.time() - start_time < timeout:
            action = self.get_action(obs, 0)
            obs, reward, done, info = env.step(action)

        return info if time.time() - start_time < timeout else "timeout"

    def train(self, env, agent_info, seconds=None, max_steps=None, copy_freq=200000):
        training_start = time.time()
        saving_time = 0

        steps = 0

        if self.experience_replay:
            self.buffer = ReplayBuffer(self.buffer_size)
            for obs, action, reward, obs2, step in get_random_experience(env, total=self.buffer_size // 5):
                self.buffer.add(obs, action, reward, obs2, step)

        obs = env.reset()
        while True:
            a = self.get_action(obs, self.epsilon)
            obs2, reward, done, _ = env.step(a)
            if self.experience_replay:
                self.buffer.add(obs, a, reward, obs2, steps)
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)
            obs = obs2 if not done else env.reset()

            if steps % copy_freq == 0 and self.dir is not None:
                self.save(time.time() - training_start, steps, env.nfeatures, extra_info=agent_info)

            if self.fixed_q_target and steps % self.reset_target_freq == 0:
                self.reset_target(env.nfeatures)

            steps += 1

            if seconds is not None and time.time() - training_start - saving_time < seconds:
                break

            if max_steps is not None and steps >= max_steps:
                break

        if self.dir is not None:
            self.save(time.time() - training_start, steps, env.nfeatures, extra_info=agent_info)

    # Takes action according to self.model
    def get_action(self, actionFeatures, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(len(actionFeatures))
        else:
            return np.argmax(self.eval_model(actionFeatures))

    def eval_model(self, actionFeatures):
        if not self.has_learned_something or actionFeatures is None:
            if self.verbose:
                print("Model evaluation is 0", self.has_learned_something, actionFeatures is None)
            return 0
        return self.model.predict(actionFeatures)

    def eval_target(self, actionFeatures):
        if not self.has_learned_something or actionFeatures is None:
            if self.verbose:
                print("Target evaluation is 0", self.has_learned_something, actionFeatures is None)
            return 0
        return self.target.run(None, {'X': actionFeatures})[0]

    def update(self, obs, action, reward, obs2):
        value = np.max(self.eval_target(obs2) if self.fixed_q_target else self.eval_model(obs2))
        self.model.partial_fit([obs[action]], [value+reward])
        self.has_learned_something = True
        if self.verbose:
            print("Normal update. Value:", value+reward)

    def batch_update(self):
        obses, actions, rewards, obses2, steps = self.buffer.sample(self.batch_size)
        if not self.fixed_q_target:
            values = np.array([np.max(self.eval_model(state)) for state in obses2])
        else:
            values = np.array([np.max(self.eval_target(state)) for state in obses2])
        if self.verbose:
            print("Batch update. Values:", rewards+values, "Steps:", steps)
        self.model.partial_fit([obses[i][actions[i]] for i in range(len(actions))], rewards + values)
        self.has_learned_something = True

    def save(self, training_time, steps, nfeatures, extra_info=None):
        os.makedirs(self.dir, exist_ok=True)
        X_test = np.array([[0 for _ in range(nfeatures)]]).astype(np.float32)
        onx = to_onnx(self.model, X_test)
        onnx.save(onx, self.dir + "/" + str(self.save_idx) + ".onnx")

        with open(self.dir + "/" + str(self.save_idx) + ".json", "w") as f:
            info = {
                "training time": training_time,
                "training steps": steps,
                "eta": self.eta,
                "nnsize": self.nnsize,
                "epsilon": self.epsilon,
                "nfeatures": nfeatures,
                "target q": self.fixed_q_target,
                "reset target freq": self.reset_target_freq,
                "experience replay": self.experience_replay,
                "buffer size": self.buffer_size
            }
            info.update(extra_info if extra_info is not None else {})
            json.dump(info, f)

        print("Agent", self.save_idx, "saved. Training time:", training_time)
        self.save_idx += 1

    def reset_target(self, nfeatures):
        if self.verbose:
            print("Resetting target.")
        X_test = np.array([[0 for _ in range(nfeatures)]]).astype(np.float32)
        self.target = InferenceSession(to_onnx(self.model, X_test).SerializeToString())
