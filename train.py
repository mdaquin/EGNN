import torch
from egnn.dataset import EGNNDataset
from torch_geometric.loader import DataLoader


train_dataset = torch.load("data/train.pt", weights_only=False)
test_dataset = torch.load("data/test.pt", weights_only=False)

print(f'Number of train graphs: {len(train_dataset)}')
print(f'First graph:{train_dataset[0]}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f'First graph:{test_dataset[0]}')

torch.manual_seed(42) 

# TODO: batch size as option
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# should be a torch_geometric dataset, not 
# a basic torch dataset. Creating the data should include
# adding Data(x=x, edge_index, y=y) 
# see https://stackoverflow.com/questions/66788555/how-to-create-a-graph-neural-network-dataset-pytorch-geometric 
# for example. TODO: review create_graph_dataset
for data in train_loader:
    print(data)