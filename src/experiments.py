from synthesizers import OnTheFlyRL, OnTheFlyRandom
import copy
from agent import Agent
from environment import DCSSolverEnv
from model import TorchModel, NeuralNetwork
from util import *
import time
import pickle
import os
import json
from topology import get_compatible_autoencoder_graphnet


class Experiment:
    def __init__(self, args, problem):
        self.args = args
        self.problem = problem
        self.results_path = "./experiments/results/" + args.exp_path + "/" + self.problem + "/"
        if args.enable_autoencoder: assert(args.exploration_graph)

class TrainingExperiment(Experiment):
    def __init__(self, args, problem: str, context: tuple[int, int]):
        super().__init__(args, problem)
        self.save_features()
        self.training_contexts = [context]
        self.env = self.init_envs()
        breakpoint()
        self.autoencoder, self.latent_space_dim = get_compatible_autoencoder_graphnet(problem, context, enabled = args.enable_autoencoder)
        self.nfeatures = self.env[context].javaEnv.getNumberOfFeatures() + int(args.enable_autoencoder) * self.latent_space_dim
        self.problem = problem
        self.agent = self.init_agent()
        self.partially_trained = False

    def init_envs(self):
        envs = {}
        for instance in self.training_contexts:
            n, k = instance
            envs[instance] = DCSSolverEnv(self.problem, n, k, self.results_path + "features.txt", exploration_graph=self.args.exploration_graph)
        return envs

    def init_agent(self):
        nn_model = self.default_network()  # customizable
        agent = Agent(self.args, save_file=self.results_path, verbose=False, nn_model=nn_model)
        self.write_description()
        return agent

    def save_features(self):
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        f = open(self.results_path + "features.txt", "w")
        features = ["ra", "labels", "context", "state_labels", "je",
                    "nk", "prop", "visits", "ltr", "boolean", "cbs"]
        for feature in features:
            f.write(feature + " " + str(int(vars(self.args)[feature])) + "\n")
        f.close()

    def default_network(self):
        nn = NeuralNetwork(self.nfeatures, self.args.nn_size).to("cpu")
        nn_model = TorchModel(self.args, self.nfeatures, network=nn)
        return nn_model

    def run(self):
        assert self.args.overwrite or not os.path.exists(self.results_path + "finished.txt"), \
            "Experiment is already fully trained, training would override" \
            " previous agents."

        self.partially_trained = True
        self.print_training_characteristics()

        self.args.nfeatures = self.nfeatures

        if self.args.exp_replay:
            self.agent.initializeBuffer(self.env)

        self.agent.train(self.env[self.training_contexts[0]],
                         max_steps=self.args.training_steps,
                         save_freq=self.args.save_freq,
                         save_at_end=True,
                         early_stopping=self.args.early_stopping,
                         results_path=self.results_path)

        with open(self.results_path + "training_data.pkl", "wb") as f:
            pickle.dump((self.agent.training_data, self.args, self.env[self.training_contexts[0]].info), f)

        self.flag_as_fully_trained()

    def write_description(self):
        if not os.path.exists(self.results_path): os.makedirs(self.results_path)
        with open(self.results_path + "description.txt", 'w') as f:
            f.write("Description: " + str(self.args.desc) + "\n")
            f.write("Params: \n" + str(self.args) + "\n")

    def print_training_characteristics(self):
        print("Starting training for instances", self.training_contexts)
        print("Number of features:", self.nfeatures)
        print("Path:", self.results_path)
        print("Params:")
        print(json.dumps(vars(self.args), indent=4))

    def flag_as_fully_trained(self):
        with open(self.results_path + "finished.txt", 'w') as f:
            f.write("Fully trained. This function should write a summary of training stats in the future.")  # FIXME


class PreSelectionTesting(Experiment):
    def __init__(self, args, problem, extrapolation_space):
        super().__init__(args, problem)
        self.extrapolation_space = extrapolation_space

    def run(self,
            timeout="10h",
            max_frontier=10000,
            solved_crit=budget_and_time,
            verbose=True):

        df = []
        agents_saved = sorted([int(f[:-5]) for f in os.listdir(self.results_path) if "onnx" in f])

        np.random.seed(0)
        tested_agents = sorted(
            np.random.choice(agents_saved, min(self.args.step_2_n, len(agents_saved)), replace=False))

        if len(agents_saved) < self.args.step_2_n:
            print("Warning:", len(agents_saved), "<", self.args.step_2_n, "agents saved.")

        self.extrapolation_space.sort()

        for i in tested_agents:
            synthesizer = OnTheFlyRL(self.results_path + str(i) + ".onnx", self.results_path + "features.txt",
                                     self.problem, max_frontier=max_frontier)

            agent_results = AllInstancesEvaluation(self.args, self.problem, self.extrapolation_space).run(
                synthesizer,
                solved_crit=solved_crit,
                ebudget=self.args.step_2_budget,
                timeout=timeout,
                verbose=verbose
            )

            for row in agent_results:
                row["idx"] = i

            df += agent_results

        df = pd.DataFrame(df)
        df.to_csv(self.results_path + self.args.step_2_results)

class TrainingExperimentWithGNN(TrainingExperiment):
    def __init__(self):
        raise NotImplementedError

class BestAgentEvaluation(Experiment):

    def __init__(self, args, problem, extrapolation_space):
        super().__init__(args, problem)
        self.extrapolation_space = extrapolation_space

    def get_best(self, criterion=best_generalization_agent_ebudget):
        """
        Get the best agent resulting from the experiment whose data is stored at pre_selection_testing_csv_path,
        under the specified criterion.
        "criterion" should be a function that receives a DataFrame and returns the path to the best agent
         with the agent number.
        """
        df = pd.read_csv(self.results_path + self.args.step_2_results)
        agent = criterion(df)

        return self.results_path + str(agent) + ".onnx", agent

    def run(self):
        best_agent_path, _ = self.get_best()

        synthesizer = OnTheFlyRL(best_agent_path, self.results_path + "/features.txt", self.problem)

        step_3_results = AllInstancesEvaluation(self.args, self.problem, self.extrapolation_space).run(
            synthesizer,
            ebudget=self.args.step_3_budget,
        )

        df = pd.DataFrame(step_3_results)
        df.to_csv(self.results_path + self.args.step_3_results)


class AllInstancesEvaluation(Experiment):

    def __init__(self, args, problem, extrapolation_space):
        super().__init__(args, problem)
        self.extrapolation_space = extrapolation_space

    def run(self,
            synthesizer,
            ebudget=-1,
            solved_crit=budget_and_time,
            timeout="10h",
            verbose=True):

        df_rows = []

        if verbose:
            print("Testing", synthesizer.algorithm_name(), synthesizer.heuristic_name())

        solved = set()
        for n, k in self.extrapolation_space:
            if ((n - 1, k) not in self.extrapolation_space or (n - 1, k) in solved) and \
                    ((n, k - 1) not in self.extrapolation_space or (n, k - 1) in solved):

                test_results = synthesizer.test(
                    n,
                    k,
                    ebudget=ebudget,
                    timeout=timeout,
                    verbose=verbose
                )

                df_rows.append(test_results)

                if solved_crit(df_rows[-1]):
                    solved.add((n, k))

        if verbose:
            print("Solved:", len(solved))
        return df_rows


class TestRepeatedly:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer

    def run(self, problem, n, k, times, path, timeout="10m", ebudget=-1):
        df = []
        start = time.time()
        for i in range(times):
            print("Testing", self.synthesizer, "with", problem, n, k, i, "- time: ", time.time() - start)

            r = self.synthesizer(problem).test(problem, n, k, timeout=timeout, ebudget=ebudget)
            r["idx"] = i
            df.append(r)
        df = pd.DataFrame(df)

        df.to_csv("./results/"+path)


def get_problem_labels(problem, eps=10):
    """ Used to generate the set of labels for a given parametric problem """

    def simplify(l):
        return "".join([c for c in l if c.isalpha()])

    labels = set()
    for i in range(eps):
        synthesizer = OnTheFlyRandom(problem, debug=True)
        synthesizer.test(2, 2, timeout="10m")
        df = pd.DataFrame(synthesizer.debug_output)
        for step in list(df["actions"]):
            for l in step:
                labels.add(simplify(l))

    return labels