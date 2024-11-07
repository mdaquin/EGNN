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
        del data, cR, cG, cB, a, x, distance, dx, dy, dz, cIR, cIG, cIB, cIGr
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

