import torch
import matplotlib.pyplot as plt
import copy

torch_seed = 42 
torch.manual_seed(torch_seed)
torch.cuda.manual_seed(torch_seed) 
torch.cuda.manual_seed_all(torch_seed) 
torch.cuda.empty_cache()



def denormalize(value, key, min_dict, max_dict):
    return value * (max_dict - min_dict) + min_dict
    #return value * (max_dict[key] - min_dict[key] + 1e-8) + min_dict[key]

    


def sizeofmodel (add_Fatom,add_Katom,interaction_colors,add_3P,input_features = 4):     
    if interaction_colors == True:
        edge_dimen = 8
    else:
        edge_dimen = 4    
        
    if add_Fatom == True and add_Katom == True:
        input_features = input_features + 3
    elif add_Fatom == True and add_Katom == False:
        input_features = input_features + 2 
    elif add_Fatom == False and add_Katom == True:
        input_features = input_features + 2 
    elif add_Fatom == False and add_Katom == False:
        input_features = input_features
    if  add_3P == True: 
        input_features = input_features + 12
    return edge_dimen, input_features      

def train(model, train_loader,device,criterion,optimizer, min_values, max_values,interaction_colors=True, add_Fatom =False, add_Katom = False, add_3P = False):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)#.cuda()  
        cR = data.colR.view(data.colR.size(0), -1)
        cG = data.colG.view(data.colG.size(0), -1)
        cB = data.colB.view(data.colB.size(0), -1)
        a = data.atom.view(data.atom.size(0), -1)
        
        
        if add_3P == True:
            cIRN = data.colIRN.to(torch.float).view(len(data.colIRN), -1)
            cIGN = data.colIGreenN.to(torch.float).view(len(data.colIGreenN), -1)
            cIBN = data.colIBN.to(torch.float).view(len(data.colIBN), -1)
            cIGrN = data.colIGN.to(torch.float).view(len(data.colIGN), -1)
            if add_Fatom == True and add_Katom == True: 
                f = data.fluoride.view(data.fluoride.size(0), -1)
                m = data.metal.view(data.metal.size(0), -1)
                k = data.potassium.view(data.potassium.size(0), -1)
                x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a,f,m,k)).to(torch.float32)
            if add_Fatom == False and add_Katom == True: 
                m = data.metal.view(data.metal.size(0), -1)
                k = data.potassium.view(data.potassium.size(0), -1)
                x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a,m,k)).to(torch.float32)
            if add_Fatom == True and add_Katom == False: 
                f = data.fluoride.view(data.fluoride.size(0), -1)
                m = data.metal.view(data.metal.size(0), -1)
                x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a,f,m)).to(torch.float32)
            if add_Fatom == False and add_Katom == False: 
                x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a)).to(torch.float32)      
        else:
            if add_Fatom == True and add_Katom == True: 
                f = data.fluoride.view(data.fluoride.size(0), -1)
                m = data.metal.view(data.metal.size(0), -1)
                k = data.potassium.view(data.potassium.size(0), -1)
                x = torch.hstack((cR,cG,cB,a,f,m,k)).to(torch.float32)
            if add_Fatom == False and add_Katom == True: 
                m = data.metal.view(data.metal.size(0), -1)
                k = data.potassium.view(data.potassium.size(0), -1)
                x = torch.hstack((cR,cG,cB,a,m,k)).to(torch.float32)
            if add_Fatom == True and add_Katom == False: 
                f = data.fluoride.view(data.fluoride.size(0), -1)
                m = data.metal.view(data.metal.size(0), -1)
                x = torch.hstack((cR,cG,cB,a,f,m)).to(torch.float32)
            if add_Fatom == False and add_Katom == False: 
                x = torch.hstack((cR,cG,cB,a)).to(torch.float32)    
            
        x = x.to(device)#.cuda()
        
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
            del cIR, cIG, cIB, cIGr
        else:
            edAtt = torch.hstack((distance, dx, dy, dz)).to(torch.float32)
        out = model(x, data.edge_index, data.batch, edAtt).to(device)#.cuda() 
        real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1)
        real_dn = real # denormalize(real,'dE_scaled', min_values, max_values)
        out_dn =  out # denormalize(out,'dE_scaled', min_values, max_values)
        
        loss = criterion(out_dn, real_dn)  # Compute the loss.
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()
        del data, cR, cG, cB, a, x, distance, dx, dy, dz
    return loss    
   
def test(model, loader, device,criterion,optimizer, min_values, max_values, show=False, clear=False,interaction_colors=True, add_Fatom =False, add_Katom = False, add_3P = False):
     model.eval()
     errs = None
     if show: toshow = None
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)#.cuda()  
         cR = data.colR.view(data.colR.size(0), -1)
         cG = data.colG.view(data.colG.size(0), -1)
         cB = data.colB.view(data.colB.size(0), -1)
         a = data.atom.view(data.atom.size(0), -1)
         
         if add_3P == True:
             cIRN = data.colIRN.to(torch.float).view(len(data.colIRN), -1)
             cIGN = data.colIGreenN.to(torch.float).view(len(data.colIGreenN), -1)
             cIBN = data.colIBN.to(torch.float).view(len(data.colIBN), -1)
             cIGrN = data.colIGN.to(torch.float).view(len(data.colIGN), -1)
             if add_Fatom == True and add_Katom == True: 
                 f = data.fluoride.view(data.fluoride.size(0), -1)
                 m = data.metal.view(data.metal.size(0), -1)
                 k = data.potassium.view(data.potassium.size(0), -1)
                 x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a,f,m,k)).to(torch.float32)
             if add_Fatom == False and add_Katom == True: 
                 m = data.metal.view(data.metal.size(0), -1)
                 k = data.potassium.view(data.potassium.size(0), -1)
                 x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a,m,k)).to(torch.float32)
             if add_Fatom == True and add_Katom == False: 
                 f = data.fluoride.view(data.fluoride.size(0), -1)
                 m = data.metal.view(data.metal.size(0), -1)
                 x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a,f,m)).to(torch.float32)
             if add_Fatom == False and add_Katom == False: 
                 x = torch.hstack((cR,cG,cB,cIRN,cIGN,cIBN,cIGrN,a)).to(torch.float32)      
         else:
             if add_Fatom == True and add_Katom == True: 
                 f = data.fluoride.view(data.fluoride.size(0), -1)
                 m = data.metal.view(data.metal.size(0), -1)
                 k = data.potassium.view(data.potassium.size(0), -1)
                 x = torch.hstack((cR,cG,cB,a,f,m,k)).to(torch.float32)
             if add_Fatom == False and add_Katom == True: 
                 m = data.metal.view(data.metal.size(0), -1)
                 k = data.potassium.view(data.potassium.size(0), -1)
                 x = torch.hstack((cR,cG,cB,a,m,k)).to(torch.float32)
             if add_Fatom == True and add_Katom == False: 
                 f = data.fluoride.view(data.fluoride.size(0), -1)
                 m = data.metal.view(data.metal.size(0), -1)
                 x = torch.hstack((cR,cG,cB,a,f,m)).to(torch.float32)
             if add_Fatom == False and add_Katom == False: 
                 x = torch.hstack((cR,cG,cB,a)).to(torch.float32) 
             
             
             
         x = x.to(device)#.cuda()
         
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
             del  cIR, cIG, cIB, cIGr
         else:
             edAtt = torch.hstack((distance, dx, dy, dz)).to(torch.float32)
         
         
         out = model(x, data.edge_index, data.batch, edAtt).detach()
         real = data.dE_scaled.to(torch.float32).view(len(data.dE_scaled), -1).detach()
         
         real_dn = denormalize(real,'dE_scaled', min_values, max_values)
         out_dn =  denormalize(out,'dE_scaled', min_values, max_values)
         
         err = (real_dn-out_dn).abs()
         if errs is None: errs = err
         else: errs = torch.vstack((errs, err))
         if show:
             if toshow is None: toshow = torch.hstack((real_dn,out_dn)).cpu()
             else: toshow = torch.vstack((toshow.to(device), torch.hstack((real_dn, out_dn)))).cpu()

     if show: 

         if clear: plt.clf()
         c="b" if not clear else "lightgrey"
         plt.scatter(toshow.T[0], toshow.T[1], color=c)
         plt.plot([toshow.T[0].min(), toshow.T[0].max()], [toshow.T[0].min(), toshow.T[0].max()], color="r")
         if not clear: plt.draw()
         if not clear: plt.pause(0.0001)
     del data, cR, cG, cB, a, x, distance, dx, dy, dz 
     return errs.nanmean()

