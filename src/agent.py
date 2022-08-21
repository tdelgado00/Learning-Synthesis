import json
import time

import numpy as np

import os

from replayBuffer import ReplayBuffer
from modelEvaluation import get_random_experience
from model import MLPModel, OnnxModel, TorchModel


class Agent:
    def __init__(self, nfeatures, eta=1e-5, nnsize=(20,), epsilon=0.1, dir=None, fixed_q_target=False, reset_target_freq=10000,
                 experience_replay=False, buffer_size=10000, batch_size=32, optimizer="sgd", model="sklearn", nstep=1, verbose=False):

        if model == "sklearn":
            self.model = MLPModel(nnsize, optimizer, eta)
        else:
            assert optimizer == "adam"
            self.model = TorchModel(nfeatures, nnsize)

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

        self.nstep = nstep

        self.verbose = verbose

        self.training_data = []

        self.best_training_perf = float("inf")

        self.params = {
            "eta": eta,
            "nnsize": nnsize,
            "epsilon": epsilon,
            "target q": fixed_q_target,
            "reset target freq": reset_target_freq,
            "experience replay": experience_replay,
            "buffer size": buffer_size,
            "optimizer": optimizer,
            "nstep": nstep
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

    def initializeBuffer(self, envs):
        exp_per_instance = self.buffer_size // len(envs)
        print("Initializing buffer with", exp_per_instance, "observations per instance, and", len(envs), "instances.")

        self.buffer = ReplayBuffer(self.buffer_size)
        for env in envs.values():
            random_experience = get_random_experience(env, total=exp_per_instance, nstep=self.nstep)
            for obs, action, reward, obs2 in random_experience:
                self.buffer.add(obs, action, reward, obs2)
        print("Done.")

    def train(self, env, seconds=None, max_steps=None, copy_freq=200000, last_obs=None, save_at_end=False):
        if self.training_start is None:
            self.training_start = time.time()

        steps = 0

        obs = env.reset() if (last_obs is None) else last_obs

        last_steps = []
        while True:
            a = self.get_action(obs, self.epsilon)
            last_steps.append((obs, a))

            obs2, reward, done, info = env.step(a)

            if self.experience_replay:
                if done:
                    for j in range(len(last_steps)):
                        self.buffer.add(last_steps[j][0], last_steps[j][1], -len(last_steps) + j, None)
                    last_steps = []
                else:
                    if len(last_steps) >= self.nstep:
                        self.buffer.add(last_steps[0][0], last_steps[0][1], -self.nstep, obs2)
                    last_steps = last_steps[len(last_steps) - self.nstep + 1:]
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)

            if done:
                if info["expanded transitions"] < self.best_training_perf:
                    self.best_training_perf = info["expanded transitions"]
                    print("New best!", self.best_training_perf, "Steps:", self.training_steps)
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
        obss, actions, rewards, obss2 = self.buffer.sample(self.batch_size)
        if self.target is not None:
            values = self.target.evalBatch(obss2)
        else:
            values = self.model.evalBatch(obss2)

        if self.verbose:
            print("Batch update. Values:", rewards+values)

        self.model.batch_update(np.array([obss[i][actions[i]] for i in range(len(actions))]), rewards + values)

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
