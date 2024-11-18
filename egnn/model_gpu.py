import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import max_pool_neighbor_x
from torch_geometric.data.data import Data 

class EGNN(Module):
    def __init__(self, input_features=4, hidden_channels=16, K=3,edge_dimen=4):
        super(EGNN, self).__init__()
        self.conv1 = GATv2Conv(input_features, hidden_channels, add_self_loops=False, edge_dim=edge_dimen, heads=3) 
        self.conv11 = GATv2Conv(3*hidden_channels, hidden_channels, add_self_loops=False, edge_dim=edge_dimen, heads=3) 
        self.conv12 = GATv2Conv(3*hidden_channels, hidden_channels, add_self_loops=False, edge_dim=edge_dimen, heads=3) 
        self.conv2 = GCNConv(3*hidden_channels, hidden_channels, add_self_loops=False)  
    
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)


    def forward(self, x, edge_index, batch, edge_attr):
        res = x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv11(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv12(x, edge_index, edge_attr=edge_attr)
        x = x.relu()

        d = max_pool_neighbor_x(Data(x, edge_index)) 
        x = d.x
        edge_index = d.edge_index
        res = x
        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_max_pool(x, batch) 
        res = x
        x = self.lin1(x)
        x = torch.nn.RReLU()(res+x)
        res = x
        x = self.lin2(x)
        x = torch.nn.RReLU()(res+x) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = torch.nn.RReLU()(x) 
        return x
    
    
    

    