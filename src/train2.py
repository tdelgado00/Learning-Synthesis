from agent import Agent
from environment import DCSSolverEnv
from plots import read_monolithic
from testing import test_agent, test_all_agent, test_agents_q
from util import *
import time
import pickle


def train_agent(instances, dir, features, seconds=None, total_steps=5000000,
                copy_freq=50000,
                eta=1e-5,
                first_epsilon=0.1,
                last_epsilon=0.1,
                epsilon_decay_steps=250000,
                nnsize=(20,),
                optimizer="sgd",
                model="sklearn",
                quantum_steps=10000,
                fixed_q_target=True, reset_target_freq=10000,
                experience_replay=True, buffer_size=10000, batch_size=10,
                nstep=1,
                incremental=True,
                normalize_reward=False,
                verbose=False):
    env = {}
    for instance in instances:
        problem, n, k = instance
        env[instance] = DCSSolverEnv(problem, n, k, features, normalize_reward=normalize_reward)

    print("Starting trianing for", instances)
    print("Number of features:", env[instances[0]].nfeatures)
    print("File:", dir)
    print("nn size:", nnsize)
    print("optimizer:", optimizer)
    print("Features:", features)

    dir = "experiments/results/" + filename([instances[0][0], 2, 2]) + "/" + dir if dir is not None else None

    agent = Agent(env[instances[0]].nfeatures, eta=eta, nnsize=nnsize, optimizer=optimizer, model=model,
                  first_epsilon=first_epsilon, last_epsilon=last_epsilon, epsilon_decay_steps=epsilon_decay_steps,
                  dir=dir, fixed_q_target=fixed_q_target,
                  reset_target_freq=reset_target_freq, experience_replay=experience_replay, buffer_size=buffer_size,
                  batch_size=batch_size, nstep=nstep, verbose=verbose)

    print("Agent info:")
    print(agent.params)

    if experience_replay:
        agent.initializeBuffer(env)

    if len(instances) > 1:  # training round robin
        last_obs = {}
        i = 0
        start_time = time.time()

        if not incremental:
            total = total_steps / quantum_steps
            while i < total:
                for instance in instances:
                    print("Training with", instance, "for", quantum_steps, "steps", "i =", i, "time =",
                          time.time() - start_time)
                    last_obs[instance] = agent.train(env[instance], max_steps=quantum_steps, copy_freq=copy_freq, last_obs=last_obs.get(instance))
                    i += 1
                    if i == total:
                        break
        else:
            trans = read_monolithic()[("expanded transitions", instances[0][0])]
            diffs = np.array([trans[k][n] for problem, n, k in instances])
            print("Instance difficulties are", diffs)
            counts = {i: 0 for i in instances}
            while agent.training_steps < total_steps:
                t = agent.training_steps / total_steps
                probs = diffs**(t*2-1)
                probs /= np.sum(probs)
                #if np.random.rand() < 0.01:
                #    print("Current probs are", probs, "(steps =", str(agent.training_steps)+")")
                instance = instances[np.random.choice(list(range(len(instances))), p=probs)]
                counts[instance] += 1

                last_obs[instance] = agent.train(env[instance], max_eps=1, copy_freq=copy_freq,
                                                 last_obs=last_obs.get(instance))
            print("Finished training. Instance counts were:", counts)
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
        "state labels": 1,
        "je feature": True,
        "nk feature": False,
        "prop feature": False,
        "visits feature": False
    }

    # RQ 1.5
    #for file in ["focused_1", "focused_2", "focused_3", "focused_4"]:
    #    for problem in ["AT", "BW", "CM", "DP", "TA", "TL"]:
            #train_agent([(problem, 2, 2)], file, features, nnsize=(20,))
            #train_agent(train_instances(problem, 10000), file, features, nnsize=(64, 32), verbose=False)
            #test_all_agents_generalization(problem, file, 15, "5s", 99)
    #        test_all_agent(problem, file, 15, timeout="10m", name="all_best22", selection=best_agent_2_2)
            #test_agents_q(problem, 2, 2, file, "states.pkl")
            #save_model_q_dfs(problem, 2, 2, file, "states.pkl", best_generalization_agent)
    
    for file in ["epsdec_4"]:
        for problem in ["CM", "TA", "DP", "AT", "BW", "TL"]:
            #s = 1 if problem == "CM" else 2
            #instances = [(problem, n, k) for n in range(s, s+2) for k in range(s, s+2)]
            #train_agent(instances, file, features, optimizer="sgd", model="pytorch", normalize_reward=True,
            #            first_epsilon=1, last_epsilon=0.01, epsilon_decay_steps=250000, incremental=True)
            train_agent([(problem, 2, 2)], file, features, optimizer="sgd", nnsize=(32, 32), model="pytorch",
                        first_epsilon=1, last_epsilon=0.01, epsilon_decay_steps=250000)
            test_all_agents_generalization(problem, file, 15, "5s", 99)
            test_all_agent(problem, file, 15, timeout="10m", name="all", selection=best_generalization_agent)

            #test_all_agent(problem, file, 15, timeout="10m", name="all_best22", selection=best_agent_2_2)
            #test_agents_q(problem, 2, 2, file, "states.pkl")
            #save_model_q_dfs(problem, 2, 2, file, "states.pkl", best_generalization_agent)
