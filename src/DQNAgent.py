import pickle
import time

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

BACKUP_FREQ = 60

class StopTrainingOnTimeLimit(BaseCallback):

    def __init__(self, time_limit):
        super(StopTrainingOnTimeLimit, self).__init__(verbose=0)
        self.time_limit = time_limit

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        return time.time() - self.start_time < self.time_limit

class DQNAgent:
    def __init__(self, env, model, eta, model_file=None, results_file=None):
        self.train_env = env(self, False)
        self.test_env = Monitor(env(self, True))
        self.training_time = 0
        self.model = model(eta, self.train_env)
        self.eta = eta
        self.results = []
        self.last_backup_time = None
        self.model_file = model_file
        self.results_file = results_file
        self.training_start = None

    # def train_timesteps(self, timesteps, eval_freq, eval_eps=3):
    #     assert timesteps % eval_freq == 0
    #     callback = EvalCallback(eval_freq=eval_freq, eval_env=self.test_env, verbose=0)
    #     self.model.learn(total_timesteps=timesteps, callback=callback, n_eval_episodes=eval_eps)

    def train_time(self, time_limit, eval_freq, eval_eps=3):
        self.training_start = time.time()
        callbacks = CallbackList([EvalCallback(eval_freq=eval_freq, eval_env=self.test_env),
                                  StopTrainingOnTimeLimit(time_limit)])
        self.model.learn(total_timesteps=1000000000000000, callback=callbacks, n_eval_episodes=eval_eps)
        self.training_time += time.time() - self.training_start
        self.save_to_file()

    def save_to_file(self):
        if self.model_file is not None:
            self.model.save(self.model_file)
        if self.results_file is not None:
            with open(self.results_file, "wb") as f:
                pickle.dump(self.results, f)
        print("Backup done. Training time:", self.training_time)

    def notify_results(self, info, is_test):
        info["train_timesteps"] = None
        info["model"] = "DQN"
        info["training_time"] = self.training_time + time.time() - self.training_start
        info["eta"] = self.eta
        info["test"] = is_test
        self.results.append(info)
        if self.last_backup_time is None or time.time() - self.last_backup_time > BACKUP_FREQ:
            self.save_to_file()
            self.last_backup_time = time.time()