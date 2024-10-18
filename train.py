import torch
from egnn.dataset import EGNNDataset
from egnn.model import EGNN
from torch_geometric.loader import DataLoader # type: ignore


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


# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing

model = EGNN()
print(model)

for step, data in enumerate(train_loader):
    print(f"Step {step}: {data.num_graphs} graphs")
P