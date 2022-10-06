from agent import Agent
from environment import DCSSolverEnv
from testing import test_agent, test_agent_all_instances, test_training_agents_generalization
from util import *
import time
import pickle


def train_agent(instances, file, features, seconds=None, total_steps=5000000,
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
                early_stopping=False, base_value=False,
                verbose=False):
    env = {}
    for instance in instances:
        problem, n, k = instance
        env[instance] = DCSSolverEnv(problem, n, k, features)

    print("Starting trianing for", instances)
    print("Number of features:", env[instances[0]].nfeatures)
    print("File:", file)
    print("nn size:", nnsize)
    print("optimizer:", optimizer)
    print("Features:", features)

    if file is not None:
        file = results_path(instances[0][0], file=file)

    agent = Agent(env[instances[0]].nfeatures, eta=eta, nnsize=nnsize, optimizer=optimizer, model=model,
                  first_epsilon=first_epsilon, last_epsilon=last_epsilon, epsilon_decay_steps=epsilon_decay_steps,
                  dir=file, fixed_q_target=fixed_q_target,
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
            while i < total and (not early_stopping or not agent.converged):
                for instance in instances:
                    print("Training with", instance, "for", quantum_steps, "steps", "i =", i, "time =",
                          time.time() - start_time)
                    last_obs[instance] = agent.train(env[instance], max_steps=quantum_steps, copy_freq=copy_freq, last_obs=last_obs.get(instance), early_stopping=early_stopping)
                    i += 1
                    if i == total:
                        break
        else:
            trans = read_monolithic()[("expanded transitions", instances[0][0])]
            diffs = np.array([trans[k][n] for problem, n, k in instances])
            print("Instance difficulties are", diffs)
            while agent.training_steps < total_steps and (not early_stopping or not agent.converged):
                t = agent.training_steps / total_steps
                probs = diffs**(t*2-1)
                probs /= np.sum(probs)
                instance = instances[np.random.choice(list(range(len(instances))), p=probs)]
                last_obs[instance] = agent.train(env[instance], max_eps=1, copy_freq=copy_freq,
                                                 last_obs=last_obs.get(instance), early_stopping=early_stopping)
    else:
        agent.train(env[instances[0]], seconds=seconds, max_steps=total_steps, copy_freq=copy_freq,
                    save_at_end=True, early_stopping=early_stopping)

    if file is not None:
        with open(file + "/" + "training_data.pkl", "wb") as f:
            pickle.dump((agent.training_data, agent.params, env[instances[0]].info), f)


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
        "visits feature": False,
        "only boolean": True,
    }

    for file in ["testing"]:
        for problem in ["DP", "TA", "BW", "CM", "AT", "TL"]:
            train_agent([(problem, 2, 2)], file, features, optimizer="sgd", model="pytorch",
                        first_epsilon=1, last_epsilon=0.01, epsilon_decay_steps=250000, early_stopping=True,
                        copy_freq=5000)
            test_training_agents_generalization(problem, file, 15, "5s", 100)
            test_agent_all_instances(problem, file, 15, timeout="10m", name="all", selection=best_generalization_agent)