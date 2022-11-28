import json
import time

import numpy as np

import os

from replayBuffer import ReplayBuffer
from model import MLPModel, OnnxModel, TorchModel


class Agent:
    def __init__(self, params, save_file=None, verbose=False):
        self.params = params
        if params["model"] == "sklearn":
            self.model = MLPModel(params["nnsize"], params["optimizer"], params["eta"])
        else:
            self.model = TorchModel(params["nfeatures"], params["nnsize"], params["eta"],
                                    params["momentum"], params["nesterov"])

        self.target = None
        self.buffer = None

        self.save_file = save_file
        self.save_idx = 0

        self.training_start = None
        self.training_steps = 0
        self.epsilon = params["first epsilon"]

        self.verbose = verbose

        self.training_data = []

        self.best_training_perf = {}
        self.last_best = None
        self.converged = False

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
        exp_per_instance = self.params["buffer size"] // len(envs)

        print("Initializing buffer with", exp_per_instance, "observations per instance, and", len(envs), "instances.")

        self.buffer = ReplayBuffer(self.params["buffer size"])
        for env in envs.values():
            random_experience = ReplayBuffer.get_experience_from_random_policy(env, total_steps=exp_per_instance, nstep=self.params["nstep"])
            for action_features, reward, obs2 in random_experience:
                self.buffer.add(action_features, reward, obs2)

        print("Done.")

    def train(self, env, seconds=None, max_steps=None, max_eps=None, copy_freq=200000,
              last_obs=None, early_stopping=False, save_at_end=False):
        if self.training_start is None:
            self.training_start = time.time()
            self.last_best = 0

        steps, eps = 0, 0

        epsilon_step = (self.params["first epsilon"] - self.params["last epsilon"])
        epsilon_step /= self.params["epsilon decay steps"]

        obs = env.reset() if (last_obs is None) else last_obs

        last_steps = []
        while True:
            a = self.get_action(obs, self.epsilon)
            last_steps.append(obs[a])

            obs2, reward, done, info = env.step(a)

            if self.params["experience replay"]:
                if done:
                    for j in range(len(last_steps)):
                        self.buffer.add(last_steps[j], -len(last_steps) + j, None)
                    last_steps = []
                else:
                    if len(last_steps) >= self.params["nstep"]:
                        self.buffer.add(last_steps[0], -self.params["nstep"], obs2)
                    last_steps = last_steps[len(last_steps) - self.params["nstep"] + 1:]
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)

            if done:
                instance = (env.info["problem"], env.info["n"], env.info["k"])
                if instance not in self.best_training_perf.keys() or \
                        info["expanded transitions"] < self.best_training_perf[instance]:
                    self.best_training_perf[instance] = info["expanded transitions"]
                    print("New best at instance "+str(instance)+"!", self.best_training_perf[instance], "Steps:", self.training_steps)
                    self.last_best = self.training_steps
                info.update({
                    "training time": time.time() - self.training_start,
                    "training steps": self.training_steps,
                    "instance": instance,
                    "loss": self.model.current_loss(),
                    })
                self.training_data.append(info)
                obs = env.reset()
            else:
                obs = obs2

            if self.training_steps % copy_freq == 0 and self.save_file is not None:
                self.save(env.info)

            if self.params["target q"] and self.training_steps % self.params["reset target freq"] == 0:
                if self.verbose:
                    print("Resetting target.")
                self.target = OnnxModel(self.model)

            steps += 1
            self.training_steps += 1
            if done:
                eps += 1

            if seconds is not None and time.time() - self.training_start > seconds:
                break

            if max_steps is not None and not early_stopping and steps >= max_steps:
                break

            if max_eps is not None and eps >= max_eps:
                break

            if max_steps is not None and self.training_steps > max_steps and (self.training_steps - self.last_best) / self.training_steps > 0.33:
                print("Converged since steps are", self.training_steps, "and max_steps is", max_steps, "and last best was", self.last_best)
                self.converged = True

            if early_stopping and self.converged:
                print("Converged!")
                break

            if self.epsilon > self.params["last epsilon"] + 1e-10:
                self.epsilon -= epsilon_step

        if self.save_file is not None and save_at_end:
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
        action_featuress, rewards, obss2 = self.buffer.sample(self.params["batch size"])
        if self.target is not None:
            values = self.target.evalBatch(obss2)
        else:
            values = self.model.evalBatch(obss2)

        if self.verbose:
            print("Batch update. Values:", rewards+values)

        self.model.batch_update(np.array(action_featuress), rewards + values)

    def save(self, env_info):
        os.makedirs(self.save_file, exist_ok=True)
        OnnxModel(self.model).save(self.save_file + "/" + str(self.save_idx))

        with open(self.save_file + "/" + str(self.save_idx) + ".json", "w") as f:
            info = {
                "training time": time.time() - self.training_start,
                "training steps": self.training_steps,
            }
            info.update(self.params)
            info.update(env_info)
            json.dump(info, f)

        print("Agent", self.save_idx, "saved. Training time:", time.time() - self.training_start, "Training steps:", self.training_steps)
        self.save_idx += 1
