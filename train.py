import math
import sys
from egnn.dataset import EGNNDataset
from egnn.model import EGNN
from torch_geometric.loader import DataLoader # type: ignore
import matplotlib.pyplot as plt
import torch
import copy, time

# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing

def train():
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        cR = data.colR.view(data.colR.size(0), -1)
        cG = data.colR.view(data.colG.size(0), -1)
        cB = data.colR.view(data.colB.size(0), -1)
        a = data.atom.view(data.atom.size(0), -1)
        x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
        x = x.to(device)
        out = model(x, data.edge_index, data.batch, data.distance.to(torch.float).view(len(data.distance), -1))  # Perform a single forward pass.
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
     errs = None
     if show: toshow = None
     for data in loader:  # Iterate in batches over the training/test dataset.
         cR = data.colR.view(data.colR.size(0), -1)
         cG = data.colR.view(data.colG.size(0), -1)
         cB = data.colR.view(data.colB.size(0), -1)
         a = data.atom.view(data.atom.size(0), -1)
         x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
         x=x.to(device)
         out = model(x, data.edge_index, data.batch, data.distance.to(torch.float).view(len(data.distance), -1)).detach()
         real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1).detach()
         err = (real-out).abs()
         if errs is None: errs = err
         else: errs = torch.vstack((errs, err))
         if show:
             if toshow is None: toshow = torch.hstack((real,out))
             else: toshow = torch.vstack(torch.hstack(real, out))
         #break
     # print(apes)
     # print(apes.sum(dim=0))
     if show: 
         # print(toshow)
         plt.clf()
         plt.scatter(toshow.T[0], toshow.T[1])
         plt.plot([0.0, 1.0], [0.0, 1.0], color="r")
         plt.draw()
         plt.pause(0.0001)
     return errs.nanmean()

plt.ion()
plt.show()
torch.manual_seed(42) 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("RUNNIN ON", device)

train_dataset = torch.load("data/train.pt", weights_only=False)
test_dataset = torch.load("data/test.pt", weights_only=False)

print(f'Number of train graphs: {len(train_dataset)}')
print(f'First graph:{train_dataset[0]}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f'First graph:{test_dataset[0]}')

# TODO: normalise distances... make no difference currently

# TODO: batch size as option
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
model = EGNN(hidden_channels=128, K=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # LR in params
criterion = torch.nn.MSELoss() # TODO: in params
# criterion = torch.nn.L1Loss() 

best_test = None
best_epoch = None
ttt=0
tte=0
nepoch = 1000 # in params
for epoch in range(1, nepoch+1):
    t1 = time.time()
    train()
    tt = round((time.time()-t1)*1000)
    ttt += tt
    t1 = time.time()
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader, show=True)
    te = round((time.time()-t1)*1000)
    tte += te
    if best_test is None or test_acc < best_test:
        best_test = test_acc
        best_model = copy.deepcopy(model)
        best_epoch = epoch
    print(f'Epoch: {epoch:03d} ({tt:04d}/{te:04d}), Train MAE: {train_acc:.4f}, Test MAE: {test_acc:.4f}')

print("Best MAE on test", best_test,"at",best_epoch)
print(f"Total time {round(ttt/1000):04d}s for training, {round(tte/1000):04d}s for testing")
print(f"Average time per epoch {round(ttt/nepoch):04d}ms for training, {round(tte/nepoch):04d}ms for testing")
test(best_model, test_loader, show=True)

plt.ioff()
plt.show()