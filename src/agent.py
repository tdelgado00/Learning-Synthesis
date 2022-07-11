import json
import time

import numpy as np

import os

from replayBuffer import ReplayBuffer
from modelEvaluation import get_random_experience
from model import MLPModel, OnnxModel, TorchModel


class Agent:
    def __init__(self, nfeatures, eta=1e-5, nnsize=(20,), epsilon=0.1, dir=None, fixed_q_target=False, reset_target_freq=10000,
                 experience_replay=False, buffer_size=10000, batch_size=32, optimizer="sgd", verbose=False):

        if optimizer == "RMSprop":
            self.model = TorchModel(nfeatures, nnsize)
        else:
            self.model = MLPModel(nnsize, optimizer, eta)

        self.optimizer = optimizer
        self.fixed_q_target = fixed_q_target
        self.reset_target_freq = reset_target_freq
        self.target = None

        self.experience_replay = experience_replay
        self.buffer_size = buffer_size
        self.buffer = None
        self.batch_size = batch_size

        self.eta = eta
        self.nnsize = nnsize
        self.epsilon = epsilon

        self.dir = dir
        self.save_idx = 0

        self.training_start = None
        self.training_steps = 0

        self.verbose = verbose

        self.training_data = []

        self.params = {
            "eta": eta,
            "nnsize": nnsize,
            "epsilon": epsilon,
            "target q": fixed_q_target,
            "reset target freq": reset_target_freq,
            "experience replay": experience_replay,
            "buffer size": buffer_size,
            "optimizer": optimizer
        }

    def test(self, env, timeout=30 * 60):
        start_time = time.time()

        obs = env.reset()
        done = False
        info = None

        while not done and time.time() - start_time < timeout:
            action = self.get_action(obs, 0)
            obs, reward, done, info = env.step(action)

        return info if time.time() - start_time < timeout else "timeout"

    def train(self, env, seconds=None, max_steps=None, copy_freq=200000, last_obs=None, save_at_end=False):
        if self.training_start is None:
            self.training_start = time.time()
        steps = 0

        if self.experience_replay and self.buffer is None:
            self.buffer = ReplayBuffer(self.buffer_size)
            for obs, action, reward, obs2, step in get_random_experience(env, total=self.buffer_size // 5):
                self.buffer.add(obs, action, reward, obs2, step)

        obs = env.reset() if (last_obs is None) else last_obs
        while True:
            a = self.get_action(obs, self.epsilon)
            obs2, reward, done, info = env.step(a)

            if self.experience_replay:
                self.buffer.add(obs, a, reward, obs2, self.training_steps)
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)

            if done:
                info.update({
                    "training time": time.time() - self.training_start,
                    "training steps": self.training_steps})
                self.training_data.append(info)
                obs = env.reset()
            else:
                obs = obs2

            if self.training_steps % copy_freq == 0 and self.dir is not None:
                self.save(env.info)

            if self.fixed_q_target and self.training_steps % self.reset_target_freq == 0:
                if self.verbose:
                    print("Resetting target.")
                self.target = OnnxModel(self.model)

            steps += 1
            self.training_steps += 1
            if seconds is not None and time.time() - self.training_start > seconds:
                break

            if max_steps is not None and steps >= max_steps:
                break

        if self.dir is not None and save_at_end:
            self.save(env.info)
        return obs.copy()

    # Takes action according to self.model
    def get_action(self, s, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(len(s))
        else:
            return self.model.best(s)

    def update(self, obs, action, reward, obs2):
        if self.target is not None:
            value = self.target.eval(obs2)
        else:
            value = self.model.eval(obs2)

        self.model.single_update(obs[action], value+reward)

        if self.verbose:
            print("Single update. Value:", value+reward)

    def batch_update(self):
        obses, actions, rewards, obss2, steps = self.buffer.sample(self.batch_size)

        if self.target is not None:
            values = self.target.evalBatch(obss2)
        else:
            values = self.model.evalBatch(obss2)

        if self.verbose:
            print("Batch update. Values:", rewards+values, "Steps:", steps)

        self.model.batch_update(np.array([obses[i][actions[i]] for i in range(len(actions))]), rewards + values)

    def save(self, env_info):
        os.makedirs(self.dir, exist_ok=True)
        OnnxModel(self.model).save(self.dir + "/" + str(self.save_idx))

        with open(self.dir + "/" + str(self.save_idx) + ".json", "w") as f:
            info = {
                "training time": time.time() - self.training_start,
                "training steps": self.training_steps,
            }
            info.update(self.params)
            info.update(env_info)
            json.dump(info, f)

        print("Agent", self.save_idx, "saved. Training time:", time.time() - self.training_start, "Training steps:", self.training_steps)
        self.save_idx += 1
