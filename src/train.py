from agent import Agent
from environment import DCSSolverEnv
from modelEvaluation import save_model_q_dfs
from test import test_agents, test_agents_q, test_agent
from util import *


def train_agent(problem, n, k, dir, seconds=None, max_steps=None, eta=1e-5, epsilon=0.1, nnsize=20,
                fixed_q_target=False, reset_target_freq=10000, experience_replay=False, buffer_size=10000,
                batch_size=32, copy_freq=200000, ra_feature=False, labels=False, context_features=False, verbose=False):
    env = DCSSolverEnv(problem, n, k, ra_feature, labels, context_features)
    print("Number of features:", env.nfeatures)

    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir if dir is not None else None
    agent = Agent(eta=eta, nnsize=nnsize, epsilon=epsilon, dir=dir, fixed_q_target=fixed_q_target,
                  reset_target_freq=reset_target_freq, experience_replay=experience_replay, buffer_size=buffer_size,
                  batch_size=batch_size, verbose=verbose)

    agent.train(env, {"ra feature": ra_feature, "labels": labels, "context features": context_features},
                seconds=seconds, max_steps=max_steps, copy_freq=copy_freq)

    return agent


def best_generalization_agent(problem, file):
    df = pd.read_csv("experiments/results/"+filename([problem, 2, 2])+"/"+file+"/generalization_all.csv")
    # por cada idx necesito cantidad de elementos y suma de las expanded transitions, nada más.


def test_all_agents_generalization(problem, file, up_to, timeout, max_idx=100):
    df = []
    for i in range(max_idx + 1):
        path = agent_path(problem, 2, 2, file, i)

        solved = [[False for _ in range(up_to)] for _ in range(up_to)]
        for n in range(up_to):
            for k in range(up_to):
                if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                    print("Testing agent with", problem, n + 1, k + 1)
                    df.append(test_agent(path, problem, n + 1, k + 1, timeout=timeout)[0])
                    df[-1]["idx"] = i
                    if not np.isnan(df[-1]["synthesis time(ms)"]):
                        solved[n][k] = True

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")


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
    labels = True
    context_features = True
    file = "5mill_C"

    n, k = 2, 2
    for problem in ["AT", "DP"]:
        #train_agent(problem, n, k, file, max_steps=max_steps, copy_freq=copy_freq,
        #            fixed_q_target=target, reset_target_freq=reset_target,
        #            experience_replay=replay, buffer_size=buffer_size, batch_size=batch_size,
        #            labels=labels, ra_feature=ra_feature,
        #            nnsize=nnsize, eta=eta,
        #            context_features=context_features,
        #            verbose=False)
        test_all_agents_generalization(problem, file, 15, "5s", 100)
        #test_agents(problem, n, k, problem, 2, 2, file)
        #test_agents(problem, n, k, problem, 2, 3, file)
        #test_agents(problem, n, k, problem, 3, 2, file)
        #test_agents(problem, n, k, problem, 3, 3, file)
        # test_agents_q(problem, n, k, file, "states_context.pkl")
        # save_model_q_dfs(problem, n, k, file, "states_context.pkl", last=True)
        # save_model_q_dfs(problem, n, k, file, "states_context.pkl", last=False)
