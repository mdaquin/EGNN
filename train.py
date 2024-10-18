from egnn.dataset import EGNNDataset
from egnn.model import EGNN
from torch_geometric.loader import DataLoader # type: ignore
import matplotlib.pyplot as plt
import torch
import copy

train_dataset = torch.load("data/train.pt", weights_only=False)
test_dataset = torch.load("data/test.pt", weights_only=False)

print(f'Number of train graphs: {len(train_dataset)}')
print(f'First graph:{train_dataset[0]}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f'First graph:{test_dataset[0]}')

torch.manual_seed(42) 

# TODO: batch size as option
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        cR = data.colR.view(data.colR.size(0), -1)
        cG = data.colR.view(data.colG.size(0), -1)
        cB = data.colR.view(data.colB.size(0), -1)
        a = data.atom.view(data.atom.size(0), -1)
        x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
        out = model(x, data.edge_index, data.batch)  # Perform a single forward pass.
        out[out == float("Inf")] = 0 
        real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1)
        # print(torch.hstack((out, real)))
        loss = criterion(out, real)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader, show=False):
     model.eval()
     sum=0
     apes = None
     if show: toshow = None
     for data in loader:  # Iterate in batches over the training/test dataset.
         cR = data.colR.view(data.colR.size(0), -1)
         cG = data.colR.view(data.colG.size(0), -1)
         cB = data.colR.view(data.colB.size(0), -1)
         a = data.atom.view(data.atom.size(0), -1)
         x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
         out = model(x, data.edge_index, data.batch).detach()
         real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1).detach()
         ape = (real-out).abs()/real.abs()
         if apes is None: apes = ape
         else: apes = torch.vstack((apes, ape))
         if show:
             if toshow is None: toshow = torch.hstack((real,out))
             else: toshow = torch.vstack(torch.hstack(real, out))
         #break
     # print(apes)
     # print(apes.sum(dim=0))
     if show: 
         # print(toshow)
         plt.scatter(toshow.T[0], toshow.T[1])
         plt.show()
     return apes.nanmean()

model = EGNN(hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

best_test = None
best_epoch = None
for epoch in range(1, 500):
    train()
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    if best_test is None or test_acc < best_test:
        best_test = test_acc
        best_model = copy.deepcopy(model)
        best_epoch = epoch
    print(f'Epoch: {epoch:03d}, Train MAPE: {train_acc:.4f}, Test MAPE: {test_acc:.4f}')

print("Best MAPE on test", best_test,"at",best_epoch)
test(best_model, test_loader, show=True)
