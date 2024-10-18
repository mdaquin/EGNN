from egnn.dataset import EGNNDataset
from egnn.model import EGNN
from torch_geometric.loader import DataLoader # type: ignore
import torch

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        cR = data.colR.view(data.colR.size(0), -1)
        cG = data.colR.view(data.colG.size(0), -1)
        cB = data.colR.view(data.colB.size(0), -1)
        a = data.atom.view(data.atom.size(0), -1)
        x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
        out = model(x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()
     sum=0
     apes = None
     for data in loader:  # Iterate in batches over the training/test dataset.
         cR = data.colR.view(data.colR.size(0), -1)
         cG = data.colR.view(data.colG.size(0), -1)
         cB = data.colR.view(data.colB.size(0), -1)
         a = data.atom.view(data.atom.size(0), -1)
         x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
         out = model(x, data.edge_index, data.batch).detach()
         #print("out")
         #print(out)
         real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1).detach()
         #print("real")
         #print(real)
         ape = (real-out).abs()/real.abs()
         if apes is None: apes = ape
         else: apes = torch.vstack((apes, ape))
         #break
     # print(apes)
     # print(apes.sum(dim=0))
     return apes.nanmean()

for epoch in range(1, 1000):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

