import sys
from agent import Agent
from environment import DCSSolverEnv
from testing import test_agent_all_instances, test_training_agents_generalization
from util import *
import time
import pickle
import os


def train_agent(instances,
                file,
                agent_params,
                features,
                quantum_steps=10000,
                incremental=True,
                seconds=None,
                total_steps=500000,
                copy_freq=5000,
                early_stopping=True,
                verbose=False):
    env = {}
    for instance in instances:
        problem, n, k = instance
        env[instance] = DCSSolverEnv(problem, n, k, features)

    print("Starting training for instances", instances)
    print("Number of features:", env[instances[0]].nfeatures)
    print("File:", file)
    print("Agent params:", agent_params)
    print("Features:", features)

    agent_params["nfeatures"] = env[instances[0]].nfeatures

    if file is not None:
        file = results_path(instances[0][0], file=file)

    agent = Agent(agent_params, save_file=file, verbose=verbose)

    if agent_params["experience replay"]:
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
        "labelsThatReach_feature": False,
        "only boolean": True,
    }
    agent_params = {
        "eta": 1e-5,
        "first epsilon": 1.0,
        "last epsilon": 0.01,
        "epsilon decay steps": 250000,
        "nnsize": (20,),
        "optimizer": "sgd",
        "model": "pytorch",
        "target q": True,
        "reset target freq": 10000,
        "experience replay": True,
        "buffer size": 10000,
        "batch size": 10,
        "nstep": 1,
        "momentum": 0.9,
        "nesterov": True
    }

    if len(sys.argv) != 2:
        print("A folder name to save results should be specified.")
        exit()

    problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
    for p in problems:
        if not os.path.isdir("results/" + p):
            os.makedirs("results/" + p)

    # TODO: -> transitions budget
    file = sys.argv[1]
    for p in problems:
        train_agent([(p, 2, 2)], file, agent_params, features)
        test_training_agents_generalization(p, file, 15, "10h", 100, ebudget=5000)
        test_agent_all_instances(p, file, 15, timeout="10m", name="all", selection=best_generalization_agent_ebudget ,ebudget=-1)
        test_agent_all_instances(p, file, 15, timeout="3h", name="all", selection=best_generalization_agent_ebudget, ebudget=15000)
