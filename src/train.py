from agent import Agent
from environment import DCSSolverEnv
from test import test_agents, test_agents_q
from util import *
from modelEvaluation import get_agent_q_df


def train_agent(problem, n, k, minutes, dir, eta=1e-5, epsilon=0.1, nnsize=20, fixed_q_target=False, reset_target_freq=10000, copy_freq=200000, ra_feature=False):
    env = DCSSolverEnv(problem, n, k, ra_feature)
    print("Number of features:", env.nfeatures)

    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir if dir is not None else None
    agent = Agent(eta=eta, nnsize=nnsize, epsilon=epsilon, dir=dir, fixed_q_target=fixed_q_target, reset_target_freq=reset_target_freq)

    agent.train(env, minutes * 60, copy_freq=copy_freq, agent_info={"ra feature": ra_feature})

    return agent


agent_idx = {
    "AT": 95,
    "TA": 105,
    "TL": 105,
    "BW": 95,
    "DP": 130
}

if __name__ == "__main__":

    for problem in ["AT", "DP", "TL", "TA", "CM", "BW"]:
        for file, target_q in [("ra_feature2opt_2h", True), ("ra_feature2_target", False)]:
            train_agent(problem, 2, 2, 120, file, copy_freq=50000, ra_feature=True, fixed_q_target=target_q)
            test_agents(problem, 2, 2, problem, 2, 2, file)
            test_agents(problem, 2, 2, problem, 3, 3, file, freq=5)
            test_agents_q(problem, 2, 2, file, "states_no_conflict.pkl")
