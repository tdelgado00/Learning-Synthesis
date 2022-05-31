from agent import Agent
from environment import DCSSolverEnv
from modelEvaluation import save_model_q_dfs
from test import test_agents, test_agents_q, test_agent, test_all_agent
from util import *
import time


def train_agent(problem, n, k, dir, seconds=None, max_steps=None, eta=1e-5, epsilon=0.1, nnsize=20,
                fixed_q_target=False, reset_target_freq=10000, experience_replay=False, buffer_size=10000,
                batch_size=32, copy_freq=200000, ra_feature=False, labels=False, context_features=False,
                state_labels=False,
                je_feature=False, optimizer="sgd",
                verbose=False):
    env = DCSSolverEnv(problem, n, k, ra_feature, labels, context_features, state_labels, je_feature)
    print("Starting trianing for", problem, n, k)
    print("Number of features:", env.nfeatures)
    print("File:", dir)
    print("nn size:", nnsize)
    print("optimizer:",optimizer)
    print("Features:", ra_feature, labels, context_features, state_labels, je_feature)

    dir = "experiments/results/" + filename([problem, n, k]) + "/" + dir if dir is not None else None
    agent = Agent(eta=eta, nnsize=nnsize, optimizer=optimizer, epsilon=epsilon, dir=dir, fixed_q_target=fixed_q_target,
                  reset_target_freq=reset_target_freq, experience_replay=experience_replay, buffer_size=buffer_size,
                  batch_size=batch_size, verbose=verbose)

    agent.train(env, {"ra feature": ra_feature, "labels": labels, "context features": context_features,
                      "state labels": state_labels, "je feature": je_feature},
                seconds=seconds, max_steps=max_steps, copy_freq=copy_freq, save_at_end=True)

    return agent


def train_agent_RR(instances, dir, seconds=None, max_steps=None, eta=1e-5, epsilon=0.1, nnsize=20,
                   fixed_q_target=False, reset_target_freq=10000, experience_replay=False, buffer_size=10000,
                   batch_size=32, copy_freq=200000, ra_feature=False, labels=False, context_features=False,
                   state_labels=False,
                   je_feature=False, optimizer="sgd",
                   verbose=False):
    env = {}
    for instance in instances:
        problem, n, k = instance
        env[instance] = DCSSolverEnv(problem, n, k, ra_feature, labels, context_features, state_labels, je_feature,
                                     True)

    dir = "experiments/results/" + dir if dir is not None else None
    agent = Agent(eta=eta, nnsize=nnsize, optimizer=optimizer, epsilon=epsilon, dir=dir, fixed_q_target=fixed_q_target,
                  reset_target_freq=reset_target_freq, experience_replay=experience_replay, buffer_size=buffer_size,
                  batch_size=batch_size, verbose=verbose)

    last_obs = {}
    rounds = 100
    steps = max_steps / rounds / len(instances)
    for r in range(rounds):
        print("Starting round", r)
        for instance in instances:
            print("Training with", instance, "for", steps, "steps")
            last_obs[instance] = agent.train(env[instance], {"ra feature": ra_feature, "labels": labels,
                                                             "context features": context_features,
                                                             "state labels": state_labels, "je feature": je_feature},
                                             seconds=seconds, max_steps=steps, copy_freq=copy_freq,
                                             last_obs=last_obs.get(instance))

    return agent


def test_all_agents_generalization(problem, file, up_to, timeout, max_idx=100):
    df = []
    start = time.time()
    for i in range(max_idx + 1):
        path = agent_path(problem, 2, 2, file, i)

        solved = [[False for _ in range(up_to)] for _ in range(up_to)]
        print("Testing agent", i, "with 5s timeout. Time:", time.time()-start)
        for n in range(up_to):
            for k in range(up_to):
                if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                    df.append(test_agent(path, problem, n + 1, k + 1, timeout=timeout)[0])
                    df[-1]["idx"] = i
                    if not np.isnan(df[-1]["synthesis time(ms)"]):
                        solved[n][k] = True

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")


if __name__ == "__main__":
    #max_steps = 100000
    #copy_freq = 1000
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
    state_labels = True
    je_feature=True
    optimizer="sgd"
    file = "AT RR"

    for problem in ["AT"]:  # , "BW", "CM", "DP", "TA", "TL"]:
        train_agent_RR([(problem, n, k) for n, k in [(2, 2), (2, 3), (3, 2), (3, 3)]], file, max_steps=max_steps,
                       copy_freq=copy_freq,
                       fixed_q_target=target, reset_target_freq=reset_target,
                       experience_replay=replay, buffer_size=buffer_size, batch_size=batch_size,
                       labels=labels, ra_feature=ra_feature,
                       nnsize=nnsize, eta=eta,
                       context_features=context_features,
                       je_feature=je_feature,
                       state_labels=state_labels,
                       optimizer=optimizer,
                       verbose=False)
        test_all_agents_generalization(problem, file, 15, "5s", 99)
        test_all_agent(problem, file, 15, timeout="10m", name="all")
        # test_agents_q(problem, n, k, file, "states_context.pkl")
        # save_model_q_dfs(problem, n, k, file, "states_context.pkl", last=True)
        # save_model_q_dfs(problem, n, k, file, "states_context.pkl", last=False)
