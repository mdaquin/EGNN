import torch
import matplotlib.pyplot as plt

torch_seed = 42 
torch.manual_seed(torch_seed)
torch.cuda.manual_seed(torch_seed) 
torch.cuda.manual_seed_all(torch_seed) 

def train(model, train_loader,device,criterion,optimizer,interaction_colors=True):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device).cuda()  
        cR = data.colR.view(data.colR.size(0), -1)
        cG = data.colG.view(data.colG.size(0), -1)
        cB = data.colB.view(data.colB.size(0), -1)
        a = data.atom.view(data.atom.size(0), -1)
        
        x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
        x = x.to(device).cuda()
        
        distance=data.distance.to(torch.float).view(len(data.distance), -1)
        dx = data.dx.to(torch.float).view(len(data.dx), -1)
        dy = data.dy.to(torch.float).view(len(data.dy), -1)
        dz = data.dz.to(torch.float).view(len(data.dz), -1)
        
        if interaction_colors == True:
            cIR = data.colIR.to(torch.float).view(len(data.colIR), -1)
            cIG = data.colIGreen.to(torch.float).view(len(data.colIGreen), -1)
            cIB = data.colIB.to(torch.float).view(len(data.colIB), -1)
            cIGr = data.colIG.to(torch.float).view(len(data.colIG), -1)
            edAtt = torch.hstack((distance, dx, dy, dz,cIR,cIG,cIB,cIGr)).to(torch.float32)
        else:
            edAtt = torch.hstack((distance, dx, dy, dz)).to(torch.float32)
        out = model(x, data.edge_index, data.batch, edAtt).cuda() 
        real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1)
        loss = criterion(out, real)  # Compute the loss.
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()
        
    return loss    
   
def test(model, loader, device,criterion,optimizer, show=False, clear=False,interaction_colors=True):
     model.eval()
     errs = None
     if show: toshow = None
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device).cuda()  
         cR = data.colR.view(data.colR.size(0), -1)
         cG = data.colG.view(data.colG.size(0), -1)
         cB = data.colB.view(data.colB.size(0), -1)
         a = data.atom.view(data.atom.size(0), -1)
         

         x = torch.hstack((cR,cG,cB,a)).to(torch.float32)
         x = x.to(device).cuda()
         distance=data.distance.to(torch.float).view(len(data.distance), -1)
         dx = data.dx.to(torch.float).view(len(data.dx), -1)
         dy = data.dy.to(torch.float).view(len(data.dy), -1)
         dz = data.dz.to(torch.float).view(len(data.dz), -1)
         
         if interaction_colors == True:
             cIR = data.colIR.to(torch.float).view(len(data.colIR), -1)
             cIG = data.colIGreen.to(torch.float).view(len(data.colIGreen), -1)
             cIB = data.colIB.to(torch.float).view(len(data.colIB), -1)
             cIGr = data.colIG.to(torch.float).view(len(data.colIG), -1)
             edAtt = torch.hstack((distance, dx, dy, dz,cIR,cIG,cIB,cIGr)).to(torch.float32)
         else:
             edAtt = torch.hstack((distance, dx, dy, dz)).to(torch.float32)
         
         
         out = model(x, data.edge_index, data.batch, edAtt).detach()
         real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1).detach()
         err = (real-out).abs()
         if errs is None: errs = err
         else: errs = torch.vstack((errs, err))
         if show:
             if toshow is None: toshow = torch.hstack((real,out)).cpu()
             else: toshow = torch.vstack((toshow.to(device), torch.hstack((real, out)))).cpu()

     if show: 

         if clear: plt.clf()
         c="b" if not clear else "lightgrey"
         plt.scatter(toshow.T[0], toshow.T[1], color=c)
         plt.plot([0.0, 1.0], [0.0, 1.0], color="r")
         if not clear: plt.draw()
         if not clear: plt.pause(0.0001)
     return errs.nanmean()

# =============================================================================
# plt.ion()
# plt.show()
# torch.cuda.manual_seed(42) 
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("RUNNIN ON", device)
# 
# train_dataset = torch.load("data/train_cpu.pt", weights_only=False)
# min, max = train_dataset.normalise()
# test_dataset = torch.load("data/test_cpu.pt", weights_only=False)
# test_dataset.normalise(min, max)
# 
# print(f'Number of train graphs: {len(train_dataset)}')
# print(f'First graph:{train_dataset[0]}')
# print(f'Number of test graphs: {len(test_dataset)}')
# print(f'First graph:{test_dataset[0]}')
# 
# 
# #lr=0.00005
# lr=0.001
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
# model = EGNN(hidden_channels=256, K=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# #criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss() 
# 
# best_test = None 
# best_epoch = None
# ttt=0
# tte=0
# nepoch = 1000 # in params
# for epoch in range(1, nepoch+1):
#     t1 = time.time()
#     train()
#     tt = round((time.time()-t1)*1000)
#     ttt += tt
#     t1 = time.time()
#     train_acc = test(model, train_loader, show=True, clear=True)
#     test_acc = test(model, test_loader, show=True)
#     te = round((time.time()-t1)*1000)
#     tte += te
#     if best_test is None or test_acc < best_test:
#         best_test = test_acc
#         best_model = copy.deepcopy(model)
#         best_epoch = epoch
#     print(f'Epoch: {epoch:03d} ({tt:04d}/{te:04d}), Train MAE: {train_acc:.4f}, Test MAE: {test_acc:.4f} (best: {best_test:.4f})')
# 
# print("Best MAE on test", best_test,"at",best_epoch)
# print(f"Total time {round(ttt/1000):04d}s for training, {round(tte/1000):04d}s for testing")
# print(f"Average time per epoch {round(ttt/nepoch):04d}ms for training, {round(tte/nepoch):04d}ms for testing")
# test(best_model, test_loader, show=True)
# 
# plt.ioff()
# plt.show()
# 
# =============================================================================
