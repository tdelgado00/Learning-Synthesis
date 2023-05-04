import random

import experiments
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
from main import parse_args
from environment import DCSSolverEnv


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class RandomExplorationForGCNTraining(experiments.TrainingExperiment):
    def __init__(self, args, problem: str, context: tuple[int, int]):
        print("Warning 1: this is now working exclusively on one graph being expanded. Pending round robin and curriculum expansions for GCN training. Be careful with expansions and snapshots.")
        print("Warning 2: To be debugged")
        args.exploration_graph = True
        super().__init__(args,problem,context)
        del self.agent
        del self.nfeatures

    def expand(self):

        breakpoint()
        single_environment = self.env[self.training_contexts[0]]
        rand_transition = random.choice(single_environment.get_actions())
        single_environment.step(rand_transition)


    def graphSnapshot(self):
        return self.envs[self.training_contexts[0]].explorationGraph


class GAETrainer:
    def __init__(self, gae_graphnet : torch.nn.Module, explorations : list[RandomExplorationForGCNTraining]):
        print("Warning 1: still to be implemented for parrallel-training on multiple and diverse graphs, and for non random explorations.")
        print("Warning 2: To be debugged")
        self.explorations = explorations
        self.gae = gae_graphnet
        raise NotImplementedError

    def train(self, graph_sampling_frequency, context, exploration_heuristic = "Random"):
        #idea: tomar grafos de exploracion aleatoria con cierta frecuencia, en cada muestra llevar adelante un entrenamiento del autoencoder
        raise NotImplementedError


if __name__ == "__main__":

    args = parse_args()
    breakpoint()
    example_exploration = RandomExplorationForGCNTraining(args, "AT", (2,2))

    i = 10
    while(i):
        example_exploration.expand()
        i-=1