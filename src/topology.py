import experiment
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GAETrainer:
    def __init__(self, graphnet):
        raise NotImplementedError


    def train(self, graph_sampling_frequency, context):
        #idea: tomar grafos de exploracion aleatoria con cierta frecuencia, en cada muestra llevar adelante un entrenamiento del autoencoder
        raise NotImplementedError



