from torch_geometric.data import Dataset
import torch 

class EGNNDataset(Dataset):
    def __init__(self, list_graphs): 
        self.list_graphs = list_graphs
    def __len__(self): return len(self.list_graphs)    
    def __getitem__(self, idx): return self.list_graphs[idx]
    def normalise(self, method="minmax", min=None, max=None):
        if min is None or max is None:
            min = {}
            max = {}
            for g in self.list_graphs:
                for k in g:
                    # only normalise things that are stored in 1D tensors
                    if type(k[1]) == torch.Tensor and len(k[1].shape) == 1:
                        if k[0] not in min or k[1].min() < min[k[0]]: min[k[0]] = k[1].min()
                        if k[0] not in max or k[1].max() > max[k[0]]: max[k[0]] = k[1].max()
        for g in self.list_graphs:
                for k in g:
                     if k[0] in min and k[0] in max:
                          nv = (k[1] - min[k[0]]) / (max[k[0]]-min[k[0]])
                          g.update({k[0]: nv})
        return min,max
