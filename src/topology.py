import pickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse

import networkx as nx
from torch.utils.tensorboard import SummaryWriter
import experiments
import torch
import numpy as np
import time
from torch_geometric.nn import GCNConv, GAE, Sequential
from torch import nn
from torch_geometric.utils import from_networkx
from environment import DCSSolverEnv, getTransitionType

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
        print("WARNING: Check MTSA DCSNonBlocking is set to full exploration mode if you want to build the full plant")
        self.env.reset()
        self.finished = False
        while not self.env.javaEnv.isFinished():
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
            labels = [getTransitionType(graph.get_edge_data(src, dst)['label']) for src, dst in graph.in_edges(node)]

            for i in range(len(transition_labels)):
                if transition_labels[i] in labels:
                    node_label_feature_vector[i] = 1
            graph.nodes[node]["predecessor_label_types_OHE"] = node_label_feature_vector
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

def build_full_plant_graph(problem, n, k, path):
    args = parse_args()
    G = RandomExplorationForGCNTraining(args, problem, (n, k)).full_nonblocking_random_exploration()
    pickle.dump(G, open(path + f'full_{problem}_{n}_{k}.pkl', 'wb'))
    print(f"Built {problem}_{n}_{k}")
    return G


def train(graph_path = None, model_name = "sample_graphnet"):
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

    G = None
    if graph_path is not None:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
    else:
        G = RandomExplorationForGCNTraining(args, "AT", (3, 3)).full_nonblocking_random_exploration()

    torch_graph = from_networkx(G, group_node_attrs=["features"])


    # Not used because we don't split train and test
    # data, _, _ = random_link_split.RandomLinkSplit(num_val=0.0, num_test=0.0)(torch_graph)

    data = torch_graph

    # parameters
    out_channels = 3
    num_features = data.num_features
    epochs = 100

    #the following is useful for storing the constructor image corresponding to the state_dict of the trained parameters
    graphnet_constructor_image = f"GAE(GCNEncoder({num_features},{out_channels}))"
    model = eval(graphnet_constructor_image)


    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)

    # move to GPU (if available)
    model = model.to(device)
    x = data.x.float().to(device)
    # print("Node features")

    edges = data.edge_index.to(device)
    # print("Edges")
    # print(edges)
    neg_edges = get_neg_edges(data)
    # print("Neg edges")
    # print(neg_edges)

    symmetric_edges, disconnected_edges, one_way_edges, neg_one_way_edges = get_edge_categories(data)
    print(symmetric_edges.shape, disconnected_edges.shape, one_way_edges.shape, neg_one_way_edges.shape)


    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_time = time.time()

    # print(data)

    for epoch in range(1, epochs + 1):
        loss = learn(model, optimizer, x, edges, neg_edges)

        writer.add_scalar("losses/loss", loss, epoch)
        writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)

        auc, ap = test(model,
                       x,
                       edges,
                       neg_edges
                       )

        EPS = 1e-15

        with torch.no_grad():
            z = model.encode(x.float().to(device), edges)

        # symmetric_loss = -torch.log(model.decoder(z, symmetric_edges, sigmoid=True) + EPS).mean()
        disconnected_loss = -torch.log(1 - model.decoder(z, disconnected_edges, sigmoid=True) + EPS).mean()
        one_way_loss = -torch.log(model.decoder(z, one_way_edges, sigmoid=True) + EPS).mean()
        neg_one_way_loss = -torch.log(1 - model.decoder(z, neg_one_way_edges, sigmoid=True) + EPS).mean()
        disconnected_correct = (model.decoder(z, disconnected_edges, sigmoid=True) <= 0.5).sum() / disconnected_edges.shape[1]
        one_way_correct = (model.decoder(z, one_way_edges, sigmoid=True) > 0.5).sum() / one_way_edges.shape[1]
        neg_one_way_correct = (model.decoder(z, one_way_edges, sigmoid=True) <= 0.5).sum() / neg_one_way_edges.shape[1]
        # symmetric_way_correct = (model.decoder(z, one_way_edges, sigmoid=True) > 0.5).sum()

        print("disconnected correct: ", disconnected_correct, "disconnected loss: ", disconnected_loss)
        print("one_way correct: ", one_way_correct, "one_way loss: ", one_way_loss)
        print("neg_one_way correct: ", neg_one_way_correct, "neg_one_way loss: ", neg_one_way_loss)

        writer.add_scalar("charts/AUC", auc, epoch)
        writer.add_scalar("charts/AP", ap, epoch)
        # writer.add_scalar("losses/symmetric_loss", symmetric_loss, epoch)  # Debería reducirse hacia 0
        writer.add_scalar("losses/disconnected_loss", disconnected_loss, epoch)  # Debería reducirse hacia 0
        writer.add_scalar("losses/one_way_loss", one_way_loss, epoch)  # Deberían estancarse, converge?
        writer.add_scalar("losses/neg_one_way_loss", neg_one_way_loss, epoch)  # Deberían estancarse
        writer.add_scalar("charts/disconnected_correct", disconnected_correct, epoch)
        writer.add_scalar("charts/one_way_correct", one_way_correct, epoch)
        writer.add_scalar("charts/neg_one_way_correct", neg_one_way_correct, epoch)

        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

    # print(data)
    # Z = model.encode(x, train_pos_edge_index)
    # print(Z)

    save_graphnet(graphnet_constructor_image, model, model_name)

    writer.close()
    return G, model


def save_graphnet(graphnet_constructor_image, model, model_name):
    """
    Stores at experiments/graphnets the parameters in model_name.pkl and the corresponding constructor in model_name.txt.
    """
    torch.save(model.state_dict(), "experiments/graphnets/" + model_name + ".pkl")
    file = open("experiments/graphnets/" + model_name + "_image.txt", "w")
    file.write(graphnet_constructor_image)
    file.close()

class CompostateEmbedding:
    def __init__(self, is_marked : bool, vector : np.array):
        self.is_marked = is_marked
        self.vector = vector

def plot_graph_embeddings(G: nx.DiGraph, graphnet: nn.Module):

    torch_graph = from_networkx(G, group_node_attrs=["features", "marked"], group_edge_attrs=["controllability"])
    torch_graph = torch_graph.to(device)
    x = torch_graph.x.float().to(device)
    edges = torch_graph.edge_index.to(device)
    edge_controllability = torch_graph.edge_attr.view(-1)


    featured_edge_list = [(edge, is_controllable) for (edge, is_controllable) in zip(edges.T, edge_controllability)]
    breakpoint()
    embeds = graphnet.encode((x.T[1].view(-1,1)).float().to(device), edges)
    assert embeds.shape[1]==3, "Only R^3 is plottable"
    compostate_embeds = [CompostateEmbedding(int(features[1]), embed.detach().numpy()) for (embed, features) in zip(embeds,x)]
    visualize_embeddings(compostate_embeds) # featured_edge_list





def visualize_embeddings(embeds: list[CompostateEmbedding], edge_attrs = None):
    print("Warning: edge plotting yet not supported.")
    marking_to_color = {True: "deepskyblue", False: "black"}
    x, y, z = [n.vector[0] for n in embeds], [n.vector[1] for n in embeds], [n.vector[2] for n in embeds]
    colors = [marking_to_color[n.is_marked] for n in embeds]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(x, y, z, c=colors)

    if edge_attrs is not None:
        for ea in edge_attrs:
                breakpoint()
                dx = x[ea[0][1]] - x[ea[0][0]]
                dy = y[ea[0][1]] - y[ea[0][0]]
                dz = z[ea[0][1]] - z[ea[0][0]]
                ax.plot([x[ea[0][1]], x[ea[0][0]]], [y[ea[0][1]], y[ea[0][0]]], [z[ea[0][1]], z[ea[0][0]]], color='darkblue')
                ax.quiver(x[ea[0][0]], y[ea[0][0]], z[ea[0][0]], dx, dy, dz, length=0.1, normalize=True, color='darkblue')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')

    # Show the plot
    plt.show()


# visualize_embeddings([CompostateEmbedding(True, np.array([0.5,0.5,0.5])), CompostateEmbedding(False, np.array([1,1,1]))])

def checkSolutionIsSubgraphOfPlant(problem,n,k):
    print("warning: hardcoded plant path")
    S = RandomExplorationForGCNTraining(None,problem, (n,k)).full_nonblocking_random_exploration()
    plant_name  = f"full_{problem}_{n}_{k}.pkl"
    plants_path = r'C:\Users\msoro\Desktop\Learning-Synthesis\experiments\plants'
    G_plant = None
    with open(r"C:\Users\msoro\Desktop\Learning-Synthesis\experiments\plants\full_AT_3_3.pkl", 'rb') as f:
        G_plant = pickle.load(f)

    breakpoint()



if __name__ == "__main__":
    checkSolutionIsSubgraphOfPlant("AT", 3, 3)
    #G, model_in_device = train(r"C:\Users\msoro\Desktop\Learning-Synthesis\experiments\plants\full_AT_2_2.pkl")

    #plot_graph_embeddings(G, model_in_device)