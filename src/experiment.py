from train import *
from itertools import product
from testing import test_training_agents_generalization_k_fixed
import copy
class Experiment():
    def __init__(self, folder_name : str, results_path : str, description : str):
        self.name = folder_name
        self.description = description
        self.results_path = results_path+"/"+folder_name + "/"

class PreSelectionTesting():
    "First run (test) full TrainingExperiment"
    def __init__(self, trained_agents_experiment, extrapolation_space, description):
        self.extrapolation_space = extrapolation_space
        self.trained_experiment = trained_agents_experiment
        self.description = description

    def run(self, timeout_per_problem, random_subset_size = 100, max_frontier=1000000, solved_crit=budget_and_time, ebudget=-1, verbose=True, output_file_name ="pre_selection_testing.csv", other_format = False):
        if not other_format:
            agentsPath = self.trained_experiment.results_path
            test_training_agents_generalization_k_fixed(self.trained_experiment.problem, self.trained_experiment.results_path, extrapolation_space,
                                                        timeout_per_problem, total=random_subset_size, max_frontier=max_frontier,
                                                        solved_crit=solved_crit, ebudget=ebudget, verbose=verbose, agentsPath=agentsPath,
                                                        path_to_analysis=self.trained_experiment.results_path+output_file_name, components_by_state = self.trained_experiment.features["components_by_state"])
        else:
            agentsPath = other_format["results_path"]
            test_training_agents_generalization_k_fixed(other_format["problem"],
                                                    other_format["results_path"], extrapolation_space,
                                                    timeout_per_problem, total=random_subset_size,
                                                    max_frontier=max_frontier,
                                                    solved_crit=solved_crit, ebudget=ebudget, verbose=verbose,
                                                    agentsPath=agentsPath,
                                                    path_to_analysis=other_format["results_path"] + output_file_name,
                                                    components_by_state=other_format["features"][
                                                        "components_by_state"])

class TrainingExperiment(Experiment):

    def __init__(self, folder_name : str, results_path : str, description : str, context, features):
        super().__init__(folder_name, results_path,description)
        self.training_contexts = [context]
        self.env = generateEnvironments(self.training_contexts, features)
        self.nfeatures = self.env[context].javaEnv.getNumberOfFeatures()
        self.folder_name = folder_name
        self.features = copy.deepcopy(features)
        self.agent_params = None
        self.agent = None
        self.problem = context[0]

    def init_agent(self, agent_params = agent_params):
        nn_model = self.defaultNetwork(agent_params) #customizable
        self.agent_params = agent_params
        self.agent = Agent(agent_params, save_file=self.results_path, verbose=False, nn_model=nn_model)
        self.write_description()

    def defaultNetwork(self, agent_params):
        nn_size = agent_params["nnsize"]
        nn = NeuralNetwork(self.nfeatures, nn_size).to("cpu")
        nn_model = TorchModel(self.nfeatures, agent_params["eta"],
                              agent_params["momentum"], agent_params["nesterov"], network=nn)
        return nn_model

    def run(self):
        assert(not os.path.exists(self.results_path + "finished.txt")), "Experiment is already fully trained, training would override previous agents."
        self.partially_trained = True
        train_agent(instances=self.training_contexts, pathToAgents=self.results_path, agent_params=self.agent_params, agent=self.agent,
                    env=self.env, features=self.features)
        self.flag_as_fully_trained()

    def write_description(self):
        if not os.path.exists(self.results_path): os.makedirs(self.results_path)
        with open(self.results_path + "description.txt", 'w') as f:
            f.write("Description: " + str(self.description) + "\n")
            f.write("Agent params: \n" + str(self.agent.params) + "\n")
            f.write("NN features: \n" + str(self.features))
    def flag_as_fully_trained(self):
        with open(self.results_path + "finished.txt", 'w') as f:
            f.write("Fully trained. This function should write a summary of training stats in the future.") #FIXME


"""class SingleAgentTesting(Experiment):
    "First run full PreSelectionTesting"
    raise NotImplementedError"""


if __name__ == "__main__":
    results_path = "/home/marco/Desktop/Learning-Synthesis/experiments/results/boolean_DP_2_k_over_boolean"
    description = "testing testing"


    for k in range(1,16,2):
        features["components_by_state"] = False
        training_experiment_1 = TrainingExperiment(f"boolean_DP_2_{k}_over_boolean", results_path, description,
                                                 ("DP", 2, k), features)
        training_experiment_1.init_agent(agent_params)
        training_experiment_1.run()

        time.sleep(60)

        features["components_by_state"] = True
        training_experiment_2 = TrainingExperiment(f"components_by_state_DP_2_{k}_over_boolean", results_path, description,
                                                 ("DP", 2, k), features)
        training_experiment_2.init_agent(agent_params)
        training_experiment_2.run()

        time.sleep(60)


        ns = list(range(1, 16))
        ks = [k]
        extrapolation_space = list(list(zip(ks, element))[0] for element in product(ns, repeat=len(ks)))
        extrapolation_space = [(e[1],e[0]) for e in extrapolation_space]
        #other_format = {"results_path" : results_path, "features" : features, "problem" : "DP"}
        agent_analysis_1 = PreSelectionTesting(training_experiment_1,extrapolation_space=extrapolation_space,description='agent_analysis.run("10h", random_subset_size = 100,ebudget=5000, output_file_name="first_try.csv")')

        agent_analysis_1.run("10h", random_subset_size = 100,ebudget=5000, output_file_name="100_subset_5000_budget.csv")

        time.sleep(60)

        agent_analysis_2 = PreSelectionTesting(training_experiment_2, extrapolation_space=extrapolation_space,
                                             description='agent_analysis.run("10h", random_subset_size = 100,ebudget=5000, output_file_name="first_try.csv")')

        agent_analysis_2.run("10h", random_subset_size=100, ebudget=5000, output_file_name="100_subset_5000_budget.csv")

        time.sleep(60)



