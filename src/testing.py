from test import test_agent, test_onnx, agent_path
from train import train_agent
import numpy as np


def test_java_and_python_coherent():
    print("Testing java and python coherent")
    for problem, n, k, agent_dir, agent_idx in [("AT", 1, 1, "10m_0", 95)]:#, ("AT", 2, 2, "10m_0", 95)]:
        result, debug_java = test_agent(agent_path(problem, 2, 2, "testing_base_features", 1), problem, n, k, debug=True)
        result, debug_python = test_onnx(agent_path(problem, 2, 2, "testing_base_features", 1), problem, n, k, debug=True)
        assert len(debug_java) == len(debug_python)
        for i in range(len(debug_java)):
            if not np.allclose(np.round(debug_python[i]["features"], 2), debug_java[i]["features"]):
                print(i)
                print("Python", debug_python[i]["features"])
                print("Java", debug_java[i]["features"])
            if not np.allclose(debug_python[i]["values"], debug_java[i]["values"]):
                print(i)
                print("Python", debug_python[i]["values"])
                print("Java", debug_java[i]["values"])


def test_agent_no_ra():
    print("Testing agent without RA")
    train_agent("AT", 2, 2, 0.03, "testing_base_features", copy_freq=100, ra_feature=False)
    _, _ = test_agent(agent_path("AT", 2, 2, "testing_base_features", 1), "AT", 1, 2, debug=False)

def test_agent_ra():
    print("Testing agent with RA")
    train_agent("AT", 2, 2, 0.03, "testing_ra", copy_freq=100, ra_feature=True)
    _, _ = test_agent(agent_path("AT", 2, 2, "testing_ra", 1), "AT", 1, 2, debug=False)


def tests():
    test_agent_ra()
    test_agent_no_ra()
    test_java_and_python_coherent()


if __name__ == '__main__':
    tests()