import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SimpleConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, global_sort_pool, EdgePooling
from torch_geometric.nn import max_pool_neighbor_x, avg_pool_neighbor_x
from torch_geometric.data.data import Data 

####### TODO: make it so that the node features are 1 vector
# TODO: figure out how to decide on nb channels
# TODO: try other aggregations / pooling
# TODO: try layers with edge attributes (and put type of interaction on edges?)
class EGNN(Module):
    def __init__(self, hidden_channels=16, K=3):
        super(EGNN, self).__init__()
        self.conv1 = GCNConv(7, hidden_channels) 
        self.pool = EdgePooling(hidden_channels) # not the right way...
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # TODO: check if edge features could be used... for distance
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.conv7 = GCNConv(hidden_channels, hidden_channels)
        self.conv8 = GCNConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.lin4 = Linear(hidden_channels, 1)


    def forward(self, x, edge_index, batch, edge_weights):
        res = x
        x = self.conv1(x, edge_index, edge_weight=edge_weights)
        d = max_pool_neighbor_x(Data(x, edge_index))
        x = d.x
        edge_index = d.edge_index
        res = x
        x = self.conv2(x, edge_index) 
        x = (res+x).relu()
        x = global_max_pool(x, batch) 
        res = x
        x = self.lin1(x)
        x = torch.nn.RReLU()(res+x)
        res = x
        x = self.lin3(x)
        x = torch.nn.RReLU()(res+x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        x = torch.nn.RReLU()(x) 
        return x