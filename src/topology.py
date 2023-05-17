import random

import os
import argparse

from torch.utils.tensorboard import SummaryWriter
import experiments
import torch
import numpy as np
import time
from torch_geometric.nn import GCNConv, GAE, Sequential
from torch import nn
from torch_geometric.utils import train_test_split_edges, from_networkx
from environment import DCSSolverEnv, getTransitionType
from torch_geometric.transforms import random_link_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomExplorationForGCNTraining:
    def __init__(self, args, problem: str, context: tuple[int, int]):
        print(
            "Warning 1: this is now working exclusively on one graph being expanded. Pending "
            "round robin and curriculum expansions for GCN training. Be careful with expansions"
            " and snapshots.")
        print("Warning2: Initial state feature set to unmarked by default")
        self.context = context
        n, k = context
        self.env = DCSSolverEnv(problem, n, k,
                                "default_features.txt",
                                exploration_graph=True)
        self.env.reset()
        self.env.set_transition_types()
        self.problem = problem
        self.finished = None

    def graph_snapshot(self):
        return self.env.exploration_graph

    def full_nonblocking_random_exploration(self):
        print("Warning: Exploration finishes with verdict. Full plant construction pending.")
        self.env.reset()
        self.finished = False
        while not self.finished:
            self.expand()

        return self.graph_snapshot()

    def expand(self):
        rand_transition = random.randint(0, self.env.javaEnv.frontierSize() - 1)
        res = self.env.step(rand_transition)
        if res[0] is None:
            self.finished = True
        final_graph = self.graph_snapshot()
        transition_labels = self.env.transition_labels
        return final_graph
        # return self.set_neighborhood_label_features(final_graph, transition_labels)

    def set_neighborhood_label_features(self, graph, transition_labels):
        transition_labels = list(transition_labels)
        for node in graph.nodes():
            node_label_feature_vector = [0] * len(transition_labels)
            labels = [getTransitionType(graph.get_edge_data(src, dst)['label']) for src, dst in graph.out_edges(node)]

            for i in range(len(transition_labels)):
                if transition_labels[i] in labels:
                    node_label_feature_vector[i] = 1
            graph.nodes[node]["successor_label_types_OHE"] = node_label_feature_vector
        return graph

    def random_exploration(self, full=False):
        raise NotImplementedError


# def labelList()
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        planes = 128
        self.first_layer = Sequential('x, edge_index', [
            (GCNConv(in_channels, planes), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
        ])

        def conv_block():
            return Sequential('x, edge_index', [
                (GCNConv(planes, planes), 'x, edge_index -> x'),
                nn.ReLU(inplace=True)
            ])

        self.conv_blocks = Sequential('x, edge_index', [(conv_block(), 'x, edge_index -> x') for _ in range(7)])

        self.last_linear = nn.Linear(planes, out_channels)

    def forward(self, x, edge_index):
        x = self.first_layer(x, edge_index)
        x = self.conv_blocks(x, edge_index)
        x = self.last_linear(x)
        return x


# Not used yet
class GraphGenerator:
    """Ideas de testing y analisis de la performance del autoencoder"""

    def __init__(self, ):
        raise NotImplementedError


# Not used yet
class GAETrainer:
    def __init__(self, gae_graphnet: torch.nn.Module, explorations: list[RandomExplorationForGCNTraining]):
        print(
            "Warning 1: still to be implemented for parrallel-training on multiple and diverse graphs, "
            "and for non random explorations.")
        print("Warning 2: To be debugged")
        self.explorations = explorations
        self.gae = gae_graphnet
        raise NotImplementedError

    def train(self, graph_sampling_frequency, context, exploration_heuristic="Random"):
        # idea: tomar grafos de exploracion aleatoria con cierta frecuencia, en cada muestra llevar adelante un entrenamiento del autoencoder
        # for exploration in self.explorations:
        raise NotImplementedError

    def trainOnFirstFullExploration(self):
        graph = self.explorations[0].full_nonblocking_random_exploration()
        raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")

    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="The learning rate of the optimizer")

    parser.add_argument("--seed", type=int, default=2,
                        help="seed of the experiment")

    parser.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")

    args = parser.parse_args()
    return args


def test(model, x, edges, neg_edges):
    model.eval()
    with torch.no_grad():
        z = model.encode(x.float().to(device), edges)
    return model.test(z, edges, neg_edges)


def learn(model, optimizer, x, edges):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x.float().to(device), edges)
    loss = model.recon_loss(z, edges)
    # if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def get_neg_edges(data):
    n = len(data.x)
    edges = data.edge_index.T.tolist()

    neg_edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                if [i, j] not in edges:
                    neg_edges.append((i, j))

    return torch.tensor(neg_edges).T

def train():
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    G = RandomExplorationForGCNTraining(args, "AT", (3, 3)).full_nonblocking_random_exploration()
    torch_graph = from_networkx(G, group_node_attrs=["features"])

    # Not used because we don't split train and test
    # data, _, _ = random_link_split.RandomLinkSplit(num_val=0.0, num_test=0.0)(torch_graph)

    data = torch_graph

    # parameters
    out_channels = 2
    num_features = data.num_features
    epochs = 5000

    model = GAE(GCNEncoder(num_features, out_channels))

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)

    # move to GPU (if available)
    model = model.to(device)
    x = data.x.float().to(device)
    # print("Node features")
    # print(x)
    edges = data.edge_index.to(device)
    # print("Edges")
    # print(edges)
    neg_edges = get_neg_edges(data)
    # print("Neg edges")
    # print(neg_edges)

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_time = time.time()

    # print(data)

    for epoch in range(1, epochs + 1):
        loss = learn(model, optimizer, x, edges)

        writer.add_scalar("losses/loss", loss, epoch)
        writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)

        auc, ap = test(model,
                       x,
                       edges,
                       neg_edges
                       )

        writer.add_scalar("charts/AUC", auc, epoch)
        writer.add_scalar("charts/AP", ap, epoch)

        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

    # print(data)
    # Z = model.encode(x, train_pos_edge_index)
    # print(Z)

    writer.close()


if __name__ == "__main__":
    train()