import pandas as pd
from testing import test_agent
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


class AgentSelection():

    def __init__(self, info_dict, agents_path):
        self.info_dict = info_dict
        self.agents_path = agents_path
    def get_best(self, pre_selection_testing_csv_path, criterion = best_generalization_agent_ebudget):
        """
        Get the best agent resulting from the experiment whose data is stored at pre_selection_testing_csv_path, under the specified criterion.
        "criterion" should be a function that receives a DataFrame and returns the path to the best agent with the agent number.
        """
        df = pd.read_csv(pre_selection_testing_csv_path)
        agent = criterion(df)
        
        return (self.agents_path + str(agent) + ".onnx", agent)



    def test_agent(self, agent_path, agent_number, timeout, ebudget, extrapolation_space, components_by_state, debug = False):
        df_rows = []
        for problem_instance in extrapolation_space:
            results, debug = test_agent(self.agents_path,self.info_dict["problem"], problem_instance[0], problem_instance[1], agent_number = agent_number, timeout = timeout, ebudget=ebudget, components_by_state=components_by_state)
            df_rows.append(pd.Series(results))
            print(results)
            if(results['expansion_budget_exceeded']== 'true'):
                print(f"Expansion budget exceeded at {problem_instance}")
                break

        return pd.DataFrame(df_rows)

if __name__ == "__main__":
    problems = ["DP", "AT","BW", "CM", "TA", "TL"]
    for problem in problems:
        results_path_cbs = f"/home/marco/Desktop/Learning-Synthesis/experiments/results/COMPLETE_components_by_state_{problem}_2_k_over_boolean/"
        description = f"Components by state feature for {problem}, considering ALL of the component type states AND excluding components that vary with N"

        features["components_by_state"] = True
        info_dict = {"problem" : problem}
        for k in range(1,16):
            training_experiment_2 = TrainingExperiment(f"COMPLETE_components_by_state_{problem}_2_{k}_over_boolean", results_path_cbs, description,
                                                     (problem, 2, k), features)
            training_experiment_2.init_agent(agent_params)
            training_experiment_2.run()


            ns = list(range(1, 16))
            ks = [k]
            extrapolation_space = list(list(zip(ks, element))[0] for element in product(ns, repeat=len(ks)))
            extrapolation_space = [(e[1],e[0]) for e in extrapolation_space]
            #other_format = {"results_path" : results_path, "features" : features, "problem" : problem}
            time.sleep(60)

            agent_analysis_2 = PreSelectionTesting(training_experiment_2, extrapolation_space=extrapolation_space,
                                                 description='agent_analysis.run("10h", random_subset_size = 100,ebudget=5000, output_file_name="first_try.csv")')

            agent_analysis_2.run("10h", random_subset_size=100, ebudget=5000, output_file_name="100_subset_5000_budget.csv")

            time.sleep(60)
        for k in range(1, 16):
            ns = list(range(1, 16))
            ks = [k]
            extrapolation_space = list(list(zip(ks, element))[0] for element in product(ns, repeat=len(ks)))
            extrapolation_space = [(e[1], e[0]) for e in extrapolation_space]

            path_to_agents = f"/home/marco/Desktop/Learning-Synthesis/experiments/results/COMPLETE_components_by_state_{problem}_2_k_over_boolean/COMPLETE_components_by_state_{problem}_2_{k}_over_boolean/"
            path_to_agents_results_in_level_k = path_to_agents + "100_subset_5000_budget.csv"

            best_agent_testing = AgentSelection(info_dict, path_to_agents)
            (best_agent_path, best_agent_number) = best_agent_testing.get_best(pre_selection_testing_csv_path=path_to_agents_results_in_level_k)
            df_performances = best_agent_testing.test_agent(best_agent_path, agent_number=best_agent_number, timeout="10h", ebudget=15000, extrapolation_space=extrapolation_space, components_by_state = features["components_by_state"])
            df_performances.to_csv(path_to_agents + f"/15000_{best_agent_number}.csv")
            time.sleep(60)

"""if __name__ == "__main__":
    results_path_cbs = "/home/marco/Desktop/Learning-Synthesis/experiments/results/COMPLETE_components_by_state_DP_2_k_over_boolean/"
    description = "Components by state feature for DP, considering ALL of the component type states"

    features["components_by_state"] = True
    info_dict = {"problem" : "DP"}
"""
        
