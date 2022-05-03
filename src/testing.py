from test import test_agent, test_onnx, agent_path
from train import train_agent
import numpy as np


def test_java_and_python_coherent():
    print("Testing java and python coherent")
    for problem, n, k, agent_dir, agent_idx in [("AT", 2, 2, "testing", 1)]:
        for test in range(10):
            print("Test", test)
            result, debug_java = test_agent(agent_path(problem, 2, 2, "testing", 1), problem, n, k, debug=True)
            result, debug_python = test_onnx(agent_path(problem, 2, 2, "testing", 1), problem, n, k, debug=True)
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
    train_agent("AT", 2, 2, "testing", seconds=3, copy_freq=100, ra_feature=True, labels=True, experience_replay=True, fixed_q_target=True)
    _, _ = test_agent(agent_path("AT", 2, 2, "testing", 1), "AT", 2, 2, debug=False)


def test_target_and_buffer():
    problem, n, k = "AT", 2, 2
    max_steps = 100
    copy_freq = 100
    buffer_size = 5
    batch_size = 3
    reset_target = 10

    train_agent(problem, n, k, None, max_steps=max_steps, copy_freq=copy_freq, ra_feature=True, labels=True,
                context_features=True,
                fixed_q_target=True, reset_target_freq=reset_target,
                experience_replay=True, buffer_size=buffer_size, batch_size=batch_size, verbose=True)
    #test_agents(problem, n, k, problem, n, k, file)
    #test_agents_q(problem, n, k, file, "states_no_conflict.pkl")


def tests():
    test_train_agent()
    test_java_and_python_coherent()


if __name__ == '__main__':
    tests()
