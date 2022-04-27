from agent import Agent
from environment import DCSSolverEnv
from modelEvaluation import save_model_q_dfs
from test import test_agents, test_agents_q
from util import *


def train_agent(problem, n, k, dir, seconds=None, max_steps=None, eta=1e-5, epsilon=0.1, nnsize=20,
                fixed_q_target=False, reset_target_freq=10000, experience_replay=False, buffer_size=10000,
                batch_size=32, copy_freq=200000, ra_feature=False, labels=False, verbose=False):
    env = DCSSolverEnv(problem, n, k, ra_feature, labels)
    print("Number of features:", env.nfeatures)

    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir if dir is not None else None
    agent = Agent(eta=eta, nnsize=nnsize, epsilon=epsilon, dir=dir, fixed_q_target=fixed_q_target,
                  reset_target_freq=reset_target_freq, experience_replay=experience_replay, buffer_size=buffer_size,
                  batch_size=batch_size, verbose=verbose)

    agent.train(env, {"ra feature": ra_feature, "labels": labels},
                seconds=seconds, max_steps=max_steps, copy_freq=copy_freq)

    return agent


agent_idx = {
    "AT": 95,
    "TA": 105,
    "TL": 105,
    "BW": 95,
    "DP": 130
}

if __name__ == "__main__":
    max_steps = 5000000
    copy_freq = 50000
    buffer_size = 10000
    batch_size = 10
    reset_target = 10000
    target = True
    replay = True
    nnsize = 20
    eta = 1e-5
    ra_feature = True

    n, k = 2, 2
    for labels, file in [(False, "5mill_RA"), (True, "5mill_L")]:
        for problem in ["AT", "BW", "CM", "DP", "TA", "TL"]:
            train_agent(problem, n, k, file, max_steps=max_steps, copy_freq=copy_freq,
                        fixed_q_target=target, reset_target_freq=reset_target,
                        experience_replay=replay, buffer_size=buffer_size, batch_size=batch_size,
                        labels=labels, ra_feature=ra_feature,
                        nnsize=nnsize, eta=eta,
                        verbose=False)
            test_agents(problem, n, k, problem, 2, 2, file)
            test_agents(problem, n, k, problem, 3, 3, file, freq=5)
            test_agents_q(problem, n, k, file, "states_labels.pkl")
            save_model_q_dfs(problem, n, k, file, "states_labels.pkl", last=True)
            save_model_q_dfs(problem, n, k, file, "states_labels.pkl", last=False)
