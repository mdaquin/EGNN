from torch.utils.data.dataset import Dataset

class EGNNDataset(Dataset):
    def __init__(self, list_graphs): self.list_graphs = list_graphs
    def __len__(self): return len(self.list_graphs)    
    def __getitem__(self, idx): return self.list_graphs[idx]
 
