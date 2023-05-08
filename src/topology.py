import random

import experiments
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, from_networkx
from torch_geometric.nn import GAE
from main import parse_args
from environment import DCSSolverEnv
import networkx as nx
import matplotlib.pyplot as plt
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
        args.exploration_graph = True
        super().__init__(args,problem,context)
        del self.agent
        del self.nfeatures
        self.env[context].reset()
        self.finished = False
        self.single_environment = self.env[self.training_contexts[0]]
        print("Warning2: Initial state feature set to unmarked by default")
    def expand(self, seed = 9):
        random.seed(seed)
        rand_transition = random.randint(0,self.single_environment.javaEnv.frontierSize()-1)
        res = self.single_environment.step(rand_transition)
        if res[0] is None: self.finished = True


    def graphSnapshot(self):
        return self.env[self.training_contexts[0]].exploration_graph

    def full_nonblocking_random_exploration(self):
        print("Warning: Exploration finishes with veredict. Full plant construction pending.")
        while (not self.finished):
            self.expand()
        return self.graphSnapshot()



class GAETrainer:
    def __init__(self, gae_graphnet : torch.nn.Module, explorations : list[RandomExplorationForGCNTraining]):
        print("Warning 1: still to be implemented for parrallel-training on multiple and diverse graphs, and for non random explorations.")
        print("Warning 2: To be debugged")
        self.explorations = explorations
        self.gae = gae_graphnet
        raise NotImplementedError

    def train(self, graph_sampling_frequency, context, exploration_heuristic = "Random"):
        #idea: tomar grafos de exploracion aleatoria con cierta frecuencia, en cada muestra llevar adelante un entrenamiento del autoencoder
        #for exploration in self.explorations:
        raise NotImplementedError
    def trainOnFirstFullExploration(self):
        graph = self.explorations[0].full_nonblocking_random_exploration()



if __name__ == "__main__":

    args = parse_args()

    example_exploration = RandomExplorationForGCNTraining(args, "AT", (2,2))
    G = example_exploration.full_nonblocking_random_exploration()

    trainable = from_networkx(G, group_node_attrs = ["features"])

    breakpoint()
    """G = example_exploration.graphSnapshot()
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw(G, pos, with_labels=False)

    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, node_size=100)

    nx.draw_networkx_edge_labels(G, pos, ax=ax)
    ax.set_axis_off()
    breakpoint()
    plt.savefig('figure.png')
"""

