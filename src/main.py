import os
import argparse
from experiments import TrainingExperiment, PreSelectionTesting, BestAgentEvaluation


def run_current_baseline():
    args = parse_args()

    problems = args.problems.split("-")

    for problem in problems:
        TrainingExperiment(args,
                           problem,
                           (2, 2)).run()

        extrapolation_space = [(n, k) for n in range(1, args.max_instance_size+1) for k in range(1, args.max_instance_size+1)]

        PreSelectionTesting(args, problem, extrapolation_space).run()

        BestAgentEvaluation(args, problem, extrapolation_space).run()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problems", type=str,
                        help="The set of target problems: e.g. \"AT-BW-CM-DP-TA-TL\"",
                        default="AT-BW-CM-DP-TA-TL")

    parser.add_argument("--exp-path", type=str,
                        help="The path of this experiment inside results")

    parser.add_argument("--step-2-results", type=str, default="step_2_results.csv",
                        help="The filename for the results of step 2 inside the experiment path")

    parser.add_argument("--step-3-results", type=str, default="step_3_results.csv",
                        help="The filename for the results of step 3 inside the experiment path")

    parser.add_argument("--desc", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="A description for this experiment")

    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="The learning rate of the optimizer")

    parser.add_argument("--first-epsilon", type=float, default=1.0,
                        help="The initial rate of exploration of the epsilon-greedy policy")

    parser.add_argument("--last-epsilon", type=float, default=0.01,
                        help="The final rate of exploration of the epsilon-greedy policy")

    parser.add_argument("--epsilon-decay-steps", type=int, default=250000,
                        help="The number of decay steps for the rate of exploration")

    parser.add_argument("--target-q", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of Fixed Q-Target")

    parser.add_argument("--reset-target-freq", type=int, default=10000,
                        help="Number of steps between target function updates (if using Fixed Q-Targets)")

    parser.add_argument("--exp-replay", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of Experience Replay")

    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Experience Replay buffer size")

    parser.add_argument("--batch-size", type=int, default=10,
                        help="Mini-batch size (if using Experience Replay)")

    parser.add_argument("--n-step", type=int, default=1,
                        help="Lookahead size for n-step Q-Learning")

    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of Nesterov momentum with SGD")

    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")

    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="SGD weight decay")

    parser.add_argument("--training-steps", type=int, default=500000,
                        help="Steps of the training algorithm. \
                        If using early stopping these are the minimum training steps.")

    parser.add_argument("--save-freq", type=int, default=5000,
                        help="The number of training steps between each set of weights saved.")

    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of early stopping (stoppping training when no improvement"
                             " are shown after a signicant number of steps)")

    parser.add_argument("--step-2-n", type=int, default=100,
                        help="Number of agents to be tested in step 2")

    parser.add_argument("--step-2-budget", type=int, default=5000,
                        help="Expansions budget for step 2")

    parser.add_argument("--step-3-budget", type=int, default=15000,
                        help="Expansions budget for step 3")

    parser.add_argument("--ra", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the ra feature")

    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of the labels features")

    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of the context features")

    parser.add_argument("--state-labels", type=int, default=1,
                        help="Size of state labels feature history (0 to disable the feature)")

    parser.add_argument("--je", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage of the just explored feature.")

    parser.add_argument("--nk", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the nk feature.")

    parser.add_argument("--prop", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the prop feature.")

    parser.add_argument("--visits", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the visits feature.")

    parser.add_argument("--ltr", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the labelsThatReach feature.")

    parser.add_argument("--boolean", action=argparse.BooleanOptionalAction, default=True,
                        help="Toggle usage only boolean fatures.")

    parser.add_argument("--cbs", action=argparse.BooleanOptionalAction, default=False,
                        help="Toggle usage of the components by state feature.")

    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                        help="Allows saving experiment on a path that already has results")

    parser.add_argument("--max-instance-size", type=int, default=15,
                        help="Maximum value for parameters (n, k) on steps 2 and 3")

    args = parser.parse_args()
    args.nn_size = (20,)

    return args


if __name__ == "__main__":
    run_current_baseline()