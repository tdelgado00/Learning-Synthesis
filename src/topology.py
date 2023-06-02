import pickle
import random

import os
import argparse

import networkx as nx
from torch.utils.tensorboard import SummaryWriter
from torch_geometric_signed_directed import MSGNN_link_prediction

import experiments
import torch
import numpy as np
import time
from torch_geometric.nn import GCNConv, GAE, Sequential
from torch import nn
from torch_geometric.utils import from_networkx
from environment import DCSSolverEnv, getTransitionType
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch_geometric_signed_directed.nn.directed import MagNet_link_prediction
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomExplorationForGCNTraining:
    def __init__(self, args, problem: str, context: tuple[int, int]):
        print("WARNING: Check MTSA DCSNonBlocking is set to full exploration mode if you want to build the full plant")
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

        #  while self.env.javaEnv.frontierSize() > 0:
        while not self.env.javaEnv.isFinished():
            self.expand()

        return self.graph_snapshot()

    def expand(self):
        rand_transition = random.randint(0, self.env.javaEnv.frontierSize() - 1)
        res = self.env.step(rand_transition)
        if res[0] is None:
            self.finished = True
        graph = self.graph_snapshot()
        # transition_labels = self.env.transition_labels
        # self.set_neighborhood_label_features(graph, transition_labels)
        self.set_degree_features(graph)
        return graph

    def set_neighborhood_label_features(self, graph, transition_labels):
        transition_labels = list(transition_labels)
        for node in graph.nodes():
            node_label_feature_vector = [0] * len(transition_labels)
            labels = [getTransitionType(graph.get_edge_data(src, dst)['label']) for src, dst in graph.out_edges(node)]

            for i in range(len(transition_labels)):
                if transition_labels[i] in labels:
                    node_label_feature_vector[i] = 1
            graph.nodes[node]["successor_label_types_OHE"] = node_label_feature_vector

    def set_degree_features(self, graph):
        for node in graph.nodes():
            graph.nodes[node]["degree_features"] = [graph.in_degree(node), graph.out_degree(node)]

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

    parser.add_argument("--learning-rate", type=float, default=0.0001,  # 0.001 for GAE
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


def learn(model, optimizer, x, edges, neg_edges):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x.float().to(device), edges)

    # This is not really necessary, recon_loss does it by default
    perm = torch.randperm(neg_edges.shape[1])
    neg_edges_loss = neg_edges[:, perm[:edges.shape[1]]]
    loss = model.recon_loss(z, edges, neg_edges_loss)

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


# For debugging purposes
def get_edge_categories(data):
    n = len(data.x)
    edges = data.edge_index.T.tolist()

    symmetric_edges = []
    disconnected_edges = []
    one_way_edges = []
    neg_one_way_edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                if [i, j] in edges and [j, i] in edges:
                    symmetric_edges.append((i, j))
                elif [i, j] in edges and [j, i] not in edges:
                    one_way_edges.append((i, j))
                elif [i, j] not in edges and [j, i] not in edges:
                    disconnected_edges.append((i, j))
                elif [i, j] not in edges and [j, i] in edges:
                    neg_one_way_edges.append((i, j))

    return torch.tensor(symmetric_edges).T, torch.tensor(disconnected_edges).T, \
           torch.tensor(one_way_edges).T, torch.tensor(neg_one_way_edges).T


def max_distance(G):
    return 0


def build_full_plant_graph(problem, n, k, path):
    args = parse_args()
    G = RandomExplorationForGCNTraining(args, problem, (n, k)).full_nonblocking_random_exploration()
    pickle.dump(G, open(path + f'full_{problem}_{n}_{k}.pkl', 'wb'))
    print(f"Built {problem}_{n}_{k}")
    return G


def add_edge_category_stats(model, edges, symmetric_edges, disconnected_edges, one_way_edges, neg_one_way_edges, writer, epoch):
    EPS = 1e-15

    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img,
                    edge_index=edge_index,
                    query_edges=query_edges)

    symmetric_loss = -torch.log(model.decoder(z, symmetric_edges, sigmoid=True) + EPS).mean()
    disconnected_loss = -torch.log(1 - model.decoder(z, disconnected_edges, sigmoid=True) + EPS).mean()
    one_way_loss = -torch.log(model.decoder(z, one_way_edges, sigmoid=True) + EPS).mean()
    neg_one_way_loss = -torch.log(1 - model.decoder(z, neg_one_way_edges, sigmoid=True) + EPS).mean()
    disconnected_correct = (model.decoder(z, disconnected_edges, sigmoid=True) <= 0.5).sum() / disconnected_edges.shape[1]
    one_way_correct = (model.decoder(z, one_way_edges, sigmoid=True) > 0.5).sum() / one_way_edges.shape[1]
    neg_one_way_correct = (model.decoder(z, one_way_edges, sigmoid=True) <= 0.5).sum() / neg_one_way_edges.shape[1]
    symmetric_correct = (model.decoder(z, one_way_edges, sigmoid=True) > 0.5).sum()

    writer.add_scalar("losses/symmetric_loss", symmetric_loss, epoch)  # Debería reducirse hacia 0
    writer.add_scalar("losses/disconnected_loss", disconnected_loss, epoch)  # Debería reducirse hacia 0
    writer.add_scalar("losses/one_way_loss", one_way_loss, epoch)  # Deberían estancarse, converge?
    writer.add_scalar("losses/neg_one_way_loss", neg_one_way_loss, epoch)  # Deberían estancarse
    writer.add_scalar("charts/disconnected_correct", disconnected_correct, epoch)
    writer.add_scalar("charts/one_way_correct", one_way_correct, epoch)
    writer.add_scalar("charts/neg_one_way_correct", neg_one_way_correct, epoch)


def eval_output(out, y):
    probs = torch.exp(out)[:, 1].detach().numpy()
    test_ap = average_precision_score(y.cpu(), probs)
    test_auc = roc_auc_score(y.cpu(), probs)
    test_acc = accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return test_ap, test_auc, test_acc


def directed_learn(model, optimizer, X_real, X_img, y, edge_index, query_edges):
    criterion = torch.nn.NLLLoss()
    model.train()
    out = model(X_real, X_img,
                edge_index=edge_index,
                query_edges=query_edges)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_ap, train_auc, train_acc = eval_output(out, y)
    return loss.detach().item(), train_ap, train_auc, train_acc


def directed_test(model, X_real, X_img, y, edge_index, query_edges):
    # Note: Eval has different output due to drop out
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img,
                    edge_index=edge_index,
                    query_edges=query_edges)
    return eval_output(out, y)


def evaluate_nodes(model, real, imag, edge_index):
    model.eval()
    with torch.no_grad():
        i = 0
        for cheb in model.Chebs:
            # print("Layer", i, real[:5, 0])
            # print("Layer", i, real[:5, 1])
            # print("Layer", i, imag[:5, 0])
            # print("Layer", i, imag[:5, 1])
            real, imag = cheb(real, imag, edge_index, None)
            if model.activation:
                real, imag = model.complex_relu(real, imag)
            i += 1
        # print("Layer", i, real[:5, 0])
        # print("Layer", i, real[:5, 1])
        # print("Layer", i, imag[:5, 0])
        # print("Layer", i, imag[:5, 1])
        return real, imag

    # print(real.shape, imag.shape)
    # sns.scatterplot(x=real[:, 0], y=real[:, 1])
    # plt.title("Real "+str(epochs))
    # plt.show()


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
    print("Max distance between two nodes", max_distance(G))
    torch_graph = from_networkx(G, group_node_attrs=["degree_features"])
    # Not used because we don't split train and test
    # data, _, _ = random_link_split.RandomLinkSplit(num_val=0.0, num_test=0.0)(torch_graph)

    data = torch_graph

    # model = GAE(GCNEncoder(num_features, out_channels))
    model = MSGNN_link_prediction(q=0.25, K=1, num_features=2,
                                   hidden=32, label_dim=2, layer=16).to(device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = data.x.float().to(device)
    edges = data.edge_index.to(device)
    neg_edges = get_neg_edges(data)

    print("Node features")
    print(x)
    print("Edges")
    print(edges)
    print("Neg edges")
    print(neg_edges)

    # symmetric_edges, disconnected_edges, one_way_edges, neg_one_way_edges = get_edge_categories(data)
    # print(symmetric_edges.shape, disconnected_edges.shape, one_way_edges.shape, neg_one_way_edges.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)
    start_time = time.time()

    X_real = x
    X_img = torch.zeros(x.shape)  # x.clone()  # Copying example from docs, why is this not zero?
    edge_index = edges
    query_edges = torch.concat([edges.T, neg_edges.T])
    y = torch.concat([torch.ones(edges.shape[1], dtype=torch.long), torch.zeros(neg_edges.shape[1], dtype=torch.long)])

    epochs = 5000
    for epoch in range(1, epochs + 1):
        train_neg_edges_idx = np.random.choice(np.arange(neg_edges.shape[1]), replace=False, size=edges.shape[1])
        train_neg_edges = neg_edges[:, train_neg_edges_idx]
        print(train_neg_edges.shape, edges.shape)
        train_query_edges = torch.concat([edges.T, train_neg_edges.T])
        y_train = torch.concat([torch.ones(edges.shape[1], dtype=torch.long), torch.zeros(train_neg_edges.shape[1], dtype=torch.long)])

        loss, train_ap, train_auc, train_acc = directed_learn(model, optimizer, X_real.clone(), X_img.clone(), y_train,
                                                              edge_index, train_query_edges)

        writer.add_scalar("losses/loss", loss, epoch)
        writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)

        real, imag = evaluate_nodes(model, X_real.clone(), X_img.clone(), edge_index)

        test_ap, test_auc, test_acc = directed_test(model, X_real.clone(), X_img.clone(), y, edge_index, query_edges)

        # add_edge_category_stats(model, edges, symmetric_edges,
        #                         disconnected_edges, one_way_edges, neg_one_way_edges, writer, epoch)

        writer.add_scalar("charts/AUC", test_auc, epoch)
        writer.add_scalar("charts/AP", test_ap, epoch)
        writer.add_scalar("charts/ACC", test_acc, epoch)

        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}'.format(epoch, test_auc, test_ap, test_acc))

    # print(data)
    # Z = model.encode(x, train_pos_edge_index)
    # print(Z)

    writer.close()


def save_plants():
    problems = ["AT", "BW", "CM", "DP", "TA", "TL"]
    instances = [(n,k) for n in range(1,5) for k in range(1,5)]
    graph_path = "/home/marco/Desktop/Learning-Synthesis/experiments/plants/"
    for p in problems:
        for instance in instances:
            g = build_full_plant_graph(p,instance[0], instance[1], graph_path)


if __name__ == "__main__":
    train()

