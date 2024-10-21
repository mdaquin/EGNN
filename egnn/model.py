import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SimpleConv, GraphConv
from torch_geometric.nn import global_mean_pool

# TODO: make it so that the node features are 1 vector
# TODO: figure out how to decide on nb channels
# TODO: normalise node features?
class EGNN(Module):
    def __init__(self, hidden_channels=16, K=3):
        super(EGNN, self).__init__()
        self.conv1 = GCNConv(4, hidden_channels) 
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # TODO: check if edge features could be used... for distance
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.conv7 = GCNConv(hidden_channels, hidden_channels)
        self.conv8 = GCNConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)


    def forward(self, x, edge_index, batch, edge_weights):
        res = x
        x = self.conv1(x, edge_index, edge_weight=edge_weights)
        x = x.relu()
        res = x
        x = self.conv2(x, edge_index, edge_weight=edge_weights)
        x = (res+x).relu()
        res = x
        x = self.conv3(x, edge_index, edge_weight=edge_weights)
        x = (res+x).relu()
        res = x
        x = self.conv4(x, edge_index, edge_weight=edge_weights)
        x = (res+x).relu()
        res = x
        x = self.conv5(x, edge_index, edge_weight=edge_weights)
        x = (res+x).relu()
        res = x
        x = self.conv6(x, edge_index, edge_weight=edge_weights)
        x = (res+x).relu()
        res = x
        x = self.conv7(x, edge_index, edge_weight=edge_weights)
        x = (res+x).relu()
        res = x
        x = self.conv8(x, edge_index, edge_weight=edge_weights)
        x = x.relu()
        x = (res+x).relu()

        x = global_mean_pool(x, batch) 
        # res = x
        # x = self.lin1(x)
        # x = torch.nn.RReLU()(res+x) # added
        #res = x
        #x = self.lin2(x)
        #x = torch.nn.RReLU()(res+x) # added
        # x = F.dropout(x, p=0.5, training=self.training)
        res = x
        x = self.lin3(x)
        # x = torch.nn.RReLU()(x) # added
        return x