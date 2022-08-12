from agent import Agent
from environment import DCSSolverEnv
from modelEvaluation import save_model_q_dfs
from testing import test_agent, test_all_agent, test_agents_q
from util import *
import time
import pickle


def best_generalization_agent(problem, file):
    df = pd.read_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")
    max_idx = df["idx"].max()
    solved = [0 for i in range(max_idx + 1)]
    expanded = [0 for i in range(max_idx + 1)]
    for x, cant in dict(df["idx"].value_counts()).items():
        solved[x] = cant
    for x, cant in dict(df.groupby("idx")["expanded transitions"].sum()).items():
        expanded[x] = cant
    perf = [(solved[i], expanded[i], i) for i in range(max_idx + 1)]
    return max(perf, key=lambda t: (t[0], -t[1], t[2]))[2]


def train_instances(problem, max_size=10000):
    r = monolithic_results["expanded transitions", problem]
    instances = []
    for n in range(2, 16):
        for k in range(2, 16):
            if not np.isnan(r[k][n]) and r[k][n] <= 10000:
                instances.append((problem, n, k))
    return instances


def train_agent(instances, dir, features, seconds=None, total_steps=5000000,
                copy_freq=50000,
                eta=1e-5,
                epsilon=0.1,
                nnsize=(20,),
                optimizer="sgd",
                model="sklearn",
                fixed_q_target=True, reset_target_freq=10000,
                experience_replay=True, buffer_size=10000, batch_size=10,

                verbose=False):
    env = {}
    for instance in instances:
        problem, n, k = instance
        env[instance] = DCSSolverEnv(problem, n, k, features)

    print("Starting trianing for", instances)
    print("Number of features:", env[instances[0]].nfeatures)
    print("File:", dir)
    print("nn size:", nnsize)
    print("optimizer:", optimizer)
    print("Features:", features)

    dir = "experiments/results/" + filename([instances[0][0], 2, 2]) + "/" + dir if dir is not None else None

    agent = Agent(env[instances[0]].nfeatures, eta=eta, nnsize=nnsize, optimizer=optimizer, model=model, epsilon=epsilon, dir=dir, fixed_q_target=fixed_q_target,
                  reset_target_freq=reset_target_freq, experience_replay=experience_replay, buffer_size=buffer_size,
                  batch_size=batch_size, verbose=verbose)
    
    print("Agent info:")
    print(agent.params)
    if len(instances) > 1:  # training round robin
        last_obs = {}
        steps = 10000
        i = 0
        total = total_steps / steps
        start_time = time.time()
        while i < total:
            for instance in instances:
                print("Training with", instance, "for", steps, "steps", "i =", i, "time =", time.time() - start_time)
                last_obs[instance] = agent.train(env[instance],
                                                 seconds=seconds, max_steps=steps, copy_freq=copy_freq,
                                                 last_obs=last_obs.get(instance))
                i += 1
                if i == total:
                    break
    else:
        agent.train(env[instances[0]], seconds=seconds, max_steps=total_steps, copy_freq=copy_freq,
                    save_at_end=True)

    if dir is not None:
        with open(dir + "/" + "training_data.pkl", "wb") as f:
            pickle.dump((agent.training_data, agent.params, env[instances[0]].info), f)


def test_all_agents_generalization(problem, file, up_to, timeout, max_idx=100):
    df = []
    start = time.time()
    for i in range(max_idx + 1):
        path = agent_path(filename([problem, 2, 2]) + "/" + file, i)

        solved = [[False for _ in range(up_to)] for _ in range(up_to)]
        print("Testing agent", i, "with 5s timeout. Time:", time.time() - start)
        for n in range(up_to):
            for k in range(up_to):
                if (n == 0 or solved[n - 1][k]) and (k == 0 or solved[n][k - 1]):
                    df.append(test_agent(path, problem, n + 1, k + 1, timeout=timeout)[0])
                    df[-1]["idx"] = i
                    if not np.isnan(df[-1]["synthesis time(ms)"]):
                        solved[n][k] = True
        print("Solved:", np.sum(solved))

    df = pd.DataFrame(df)
    df.to_csv("experiments/results/" + filename([problem, 2, 2]) + "/" + file + "/generalization_all.csv")


if __name__ == "__main__":
    start = time.time()
    features = {
        "ra feature": False,
        "context features": True,
        "labels": True,
        "state labels": 3,
        "je feature": True,
        "nk feature": False,
        "prop feature": True,
        "visits feature": True
    }
    for file in ["testing"]:
        for problem in ["AT", "TA", "BW", "CM", "DP", "TL"]:
            train_agent([(problem, 2, 2)], file, features, nnsize=(64, 64))
            #train_agent(train_instances(problem, 10000), file, features, nnsize=(64, 32), verbose=False)
            test_all_agents_generalization(problem, file, 15, "5s", 99)
            test_all_agent(problem, file, 15, timeout="10m", name="all", selection=best_generalization_agent)
            #test_agents_q(problem, 2, 2, file, "states.pkl")
            #save_model_q_dfs(problem, 2, 2, file, "states.pkl", best_generalization_agent)
    print("Total time:", time.time()-start)
