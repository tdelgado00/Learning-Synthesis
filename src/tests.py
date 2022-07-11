from testing import test_agent, test_onnx
from util import *
from train import train_agent, test_all_agents_generalization, test_agents_q
import time

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
    test_all_agents_generalization(problem, file, 15, "1s", 1)
    test_agents_q(problem, n, k, file, "states_prop.pkl")


def tests():
    test_training_pipeline()
    test_java_and_python_coherent()


if __name__ == "__main__":
    tests()