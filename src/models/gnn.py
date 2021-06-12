import torch
from torch import nn
from torch_geometric import nn as gnn
from .pgm import InferenceModel

class GCN(nn.Module):
    """
    Simple original GCN model that we use. 2 layers with 16 units
    """
    def __init__(self, num_features, num_hidden=16, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            num_inputs = num_features if i == 0 else num_hidden
            layers.append(gnn.GCNConv(num_inputs, num_hidden, cached=True))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(num_hidden, 1)
    def forward(self, x, edge_index):
        out = x
        for layer in self.layers:
            out = layer(out, edge_index)
            out = torch.relu(out)
        return self.linear(out).view(-1)
