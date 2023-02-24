from train import *
class Experiment():
    def __init__(self, folder_name : str, description : str):
        self.name = folder_name
        self.description = description

class PreSelectionTesting(Experiment):
    "First run (test) full TrainingExperiment"
    raise NotImplementedError

class TrainingExperiment(Experiment):

    def __init__(self, folder_name : str, description : str, context, features):
        super().__init__(folder_name,description)
        self.training_contexts = [context]
        self.env = generateEnvironments(training_contexts, features)
        self.nfeatures = env[context].javaEnv.getNumberOfFeatures()
        self.write_description(folder_name)

    def init_agent(self, agent_params = agent_params):
        nn_model = self.defaultNetwork(agent_params) #customizable
        self.agent_params = agent_params
        self.agent = Agent(agent_params, save_file=results_path(problem, file=exp_folder), verbose=False, nn_model=nn_model)

    def defaultNetwork(self, agent_params):
        nn_size = agent_params["nnsize"]
        nn = NeuralNetwork(self.nfeatures, nn_size).to("cpu")
        nn_model = TorchModel(nfeatures, agent_params["eta"],
                              agent_params["momentum"], agent_params["nesterov"], network=nn)
        return nn_model

    def run(self):
        train_agent(instances=training_contexts, file=self.folder_name, agent_params=self.agent_params, agent=self.agent,
                    env=env, features=features)
    def write_description(self, folder_name):
        raise NotImplementedError



class SingleAgentTesting(Experiment):
    "First run full PreSelectionTesting"
    raise NotImplementedError