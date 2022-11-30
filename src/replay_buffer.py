import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, action_features, reward, obs2):
        data = (action_features, reward, obs2)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        action_featuress, rewards, obss = [], [], []
        for i in idxes:
            data = self._storage[i]
            action_features, reward, obs = data
            action_featuress.append(action_features)
            rewards.append(reward)
            obss.append(obs)
        return action_featuress, np.array(rewards), obss

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def __repr__(self):
        return " - ".join([str(data[:2]) for data in self._storage])

    @staticmethod
    def get_experience_from_random_policy(env, total_steps, nstep=1):
        """ A random policy is run for total_steps steps, saving the observations in the format of the replay buffer """
        states = []
        obs = env.reset()
        steps = 0

        last_steps = []
        for i in range(total_steps):
            action = np.random.randint(len(obs))
            last_steps.append(obs[action])

            obs2, reward, done, info = env.step(action)

            if done:
                for j in range(len(last_steps)):
                    states.append((last_steps[j], -len(last_steps) + j, None))
                last_steps = []
                obs = env.reset()
            else:
                if len(last_steps) >= nstep:
                    states.append((last_steps[0], -nstep, obs2))
                last_steps = last_steps[len(last_steps) - nstep + 1:]
                obs = obs2
            steps += 1
        return states
