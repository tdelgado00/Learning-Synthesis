import os
from environment import generateEnvironments
from testing import test_agent, test_training_agents_generalization, test_agent_all_instances
#from testing import test_onnx, test_agents_q
from util import *
from train import train_agent
import time
from model import *
from agent import Agent
from shutil import rmtree


"""
def test_java_and_python_coherent():
    print("Testing java and python coherent")
    for problem, n, k, agent_dir, agent_idx in [("AT", 2, 2, "testing", 1)]:
        for test in range(10):
            print("Test", test)
            result, debug_java = test_agent(agent_path(filename([problem, 2, 2])+"/"+"testing", 1), problem, n, k, debug=True)
            result, debug_python = test_onnx(agent_path(filename([problem, 2, 2])+"/"+"testing", 1), problem, n, k, debug=True)
            assert len(debug_java) == len(debug_python)
            for i in range(len(debug_java)):
                if len(debug_python[i]["features"]) != len(debug_java[i]["features"]):
                    print("Different frontier size!", len(debug_python[i]["features"]), len(debug_java[i]["features"]), "at step", i)
                for j in range(len(debug_python[i]["features"])):
                    if not np.allclose(debug_python[i]["features"][j], debug_java[i]["features"][j]):
                        print(i, j, "Features are different")
                        print("Python", list(debug_python[i]["features"][j]))
                        print("Java", debug_java[i]["features"][j])
                    if not np.allclose(debug_python[i]["values"][j], debug_java[i]["values"][j]):
                        print(i, j, "Values are different")
                        print("Python", debug_python[i]["values"][j])
                        print("Java", debug_java[i]["values"][j])




def test_train_agent():
    print("Training agent")
    start = time.time()
    train_agent("AT", 2, 2, "testing", max_steps=1000, copy_freq=500, ra_feature=True, labels=True, context_features=True,
                state_labels=True, je_feature=True, experience_replay=True, fixed_q_target=True)
    _, _ = test_agent(agent_path(filename(["AT", 2, 2])+"/"+"testing", 1), "AT", 2, 2, debug=False)
    print(time.time() - start)


def test_training_pipeline():
    problem, n, k = "AT", 2, 2

    file = "testing"
    train_agent([(problem, n, k)], file, total_steps=100, copy_freq=100, buffer_size=5, batch_size=3,
                reset_target_freq=10)
    test_training_agents_generalization(problem, file, 15, "1s", 1)
    test_agents_q(problem, n, k, file, "states_prop.pkl")
"""

sample_params = {
        "eta": 1e-5,
        "first epsilon": 1.0,
        "last epsilon": 0.01,
        "epsilon decay steps": 250000,
        "nnsize": (20,10,5),
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
sample_features = {
        "ra feature": False,
        "context features": True,
        "labels": True,
        "state labels": 1,
        "je feature": True,
        "nk feature": False,
        "prop feature": False,
        "visits feature": False,
        "labelsThatReach_feature": True,
        "only boolean": True,
    }
class ExperimentalTester:
    """This class performs a basic white and black box tests on the core functionalities to train the agent and test it on the full benchmark"""
    #TODO: lo mas importante es que corran train_agent, test_training_agents_generalization y test_agent_all_instances y que guarden los archivos correctamente
    def __init__(self, training_contexts, modelName, agent, env):
        self.training_contexts = training_contexts
        self.modelName = modelName
        self.agent = agent
        self.env = env

    def runFullTestSuit(self):
        self.testCompleteTrainingParams()
        self.testCompleteTrainingFeatures()
        self.testSampleAgentsAreStoredCorrectly()
        self.testSampleAgentsEvaluationsAreStoredCorrectly()
        self.testSampleAgentsGeneralizationIsStoredCorrectly()
    def testCompleteTrainingParams(self):
        assert(sample_params.keys() == self.agent.params.keys())
        print("PASSED")

    def testCompleteTrainingFeatures(self):
        assert(sample_features.keys() == self.env[self.training_contexts[0]].features.keys())
        print("PASSED")

    def testSampleAgentsAreStoredCorrectly(self):
        pathToModel = results_path(self.training_contexts[0][0], file = self.modelName)
        #rmtree(pathToModel, ignore_errors=True)
        train_agent(instances = self.training_contexts, file = self.modelName, agent_params=sample_params, agent = self.agent, env = self.env, features=sample_features, total_steps=100, copy_freq=10)
        modelFolderFiles = os.listdir(pathToModel)
        modelFolderFiles = [f for f in modelFolderFiles if os.path.isfile(pathToModel + '/' + f)]
        assert(len(modelFolderFiles)>0)
        print("PASSED")
    def testSampleAgentsEvaluationsAreStoredCorrectly(self):
        pathToModel = results_path(self.training_contexts[0][0], file=self.modelName)
        pathToCsv = pathToModel + "/generalization_all"+joinAsStrings([2, "5s", 100, 100])+".csv"
        try:
            os.remove(pathToCsv)
        except FileNotFoundError:
            print("Not previously evaluated")
        test_training_agents_generalization(self.training_contexts[0][0], self.modelName, 2, "5s", total=100, ebudget=100, verbose=True)

        assert("generalization_all"+joinAsStrings([2, "5s", 100, 100])+".csv" in os.listdir(pathToModel))
        print("PASSED")

    def testSampleAgentsGeneralizationIsStoredCorrectly(self):
        pathToModel = results_path(self.training_contexts[0][0], file=self.modelName)
        pathToCsv = pathToModel + "/generalization_all.csv"
        try:
            os.remove(pathToCsv)
        except FileNotFoundError:
            print("Not previously evaluated")
        test_agent_all_instances(problem=self.training_contexts[0][0], file=self.modelName, up_to=2, timeout="5s", selection=best_generalization_agent_ebudget,ebudget=100,
                                            name="all", total=100, used_testing_timeout = "5s")
        file_name = ("all" + "_" + problem + "_" + str(2) + "_" + str(100) + "_TO:" + "5s" + ".csv")
        assert (file_name in os.listdir(pathToModel))
        print("PASSED")

    def testMaintainsConsistentPerformanceWithPreviousVersions(self):
        #TODO: problema es que hay cosas del algoritmo que estan aleatorizados (ej: parametros de la red)
        pass

    def testTrainingDeviceIsCorrect(self):
            pass
            # assert(self.agent.model.device == )

    def testInferenceDeviceIsCorrect(self):
            pass

    def testNetworkIsCorrectlyInitialized(self):
            pass


def tests():
    pass
    #test_training_pipeline()
    #test_java_and_python_coherent()


if __name__ == "__main__":

    print("Insert your experimental variation here to fast-test the full training and evaluation pipeline. Starts in 5 seconds...")
    time.sleep(5)
    for problem in ["AT", "BW", "CM", "DP", "TA", "TL"]:
        exp_folder = "testSampleName"
        context = (problem, 2, 2)
        training_contexts = [context]
        env = generateEnvironments(training_contexts, sample_features)
        nfeatures = env[context].javaEnv.getNumberOfFeatures()
        nn_size = sample_params["nnsize"]
        if torch.cuda.is_available():
            time.sleep(2)
            print("Using GPU")
            time.sleep(4)
        nn = NeuralNetwork(nfeatures, nn_size).to("cuda" if torch.cuda.is_available() else "cpu")
        nn_model = TorchModel(nfeatures, sample_params["eta"],
                              sample_params["momentum"], sample_params["nesterov"], network=nn)

        agent = Agent(sample_params, save_file=results_path(problem,file = exp_folder), verbose=False, nn_model=nn_model)


        tester = ExperimentalTester(training_contexts, exp_folder, agent, env)
        tester.runFullTestSuit()
    #tests()