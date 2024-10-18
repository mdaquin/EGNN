import torch
from torch.nn import Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

# TODO: make it so that the node features are 1 vector
# TODO: figure out how to decide on nb channels
# TODO: normalise node features?
class EGNN(Module):
    def __init__(self):
        super(EGNN, self).__init__()
        self.conv1 = GCNConv(4, 16) # 4 because of the 4 node features 
        self.conv2 = GCNConv(16, 16) # TODO: check if edge features could be used... for distance
        self.conv3 = GCNConv(16, 16)
        self.lin = Linear(16, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        # x = x.sigmoid()
        x = self.conv2(x, edge_index)
        # x = x.sigmoid()
        x = self.conv3(x, edge_index)
        # x = x.sigmoid()

        x = global_mean_pool(x, batch) 

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        # print(x)
        x = x.sigmoid() # added
        # print(x)
        return x