#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:45:03 2024

@author: lklochko
"""

import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch_geometric.loader import DataLoader # type: ignore
import numpy as np
from train_gpu import test
import json 

def denormalize(value, key, min_dict, max_dict):
    return value * (max_dict - min_dict) + min_dict

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
     return errs.nanmean(),toshow.T[0],toshow.T[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("RUNNIN ON", device)

#with open("input_config_TTT.json") as f:
with open("input_config_TFF_3PF.json") as f:
   params = json.load(f)


interaction_colors = params['interaction_colors']   
learning_rate      = params['learning_rate']
batch_size_train   = params['batch_size_train']
batch_size_test    = params['batch_size_test']   
add_Fatom          = params['add_Fatom']
add_Katom          = params['add_Katom']
add_3P             = params['add_3P']
hidden_channels    = params['hidden_channels']
nRuns              = params['Number_of_RUNS']
nepoch             = params['Epochs']


# =============================================================================
# batch_size_test    = 200 
# add_Fatom          = False
# add_Katom          = False
# interaction_colors = False
# learning_rate = 0.001
# nRuns     = 10    
# =============================================================================

#df = pd.read_excel("data/data_ia_solol_kmf3.xlsx", skiprows=9, index_col=0).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)
df = pd.read_csv('data/all_new_data_corrected.csv')


minX = df["dE scaled"].min()
maxX = df["dE scaled"].max()


results = {'run': [], 'full_mae': [], 'test_mae': [], 'Energy_predicted_fullSet':[], 'Energy_target_fullSet':[], 'Energy_predicted_testSet':[],'Energy_target_testSet':[]}


for ii in range (1,nRuns+1):
    results['run'].append(ii)
    best_model = torch.load('data/best_model_%s_ic%s_F%s_K%s_3P%s.pt'%(ii,interaction_colors,add_Fatom,add_Katom,add_3P))
    optimizer = torch.optim.Adam(best_model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss() 
    
    train_dataset = torch.load("data/train_gpu_ic%s_F%s_K%s_%s_3P%s.pt"%(interaction_colors,add_Fatom,add_Katom,ii,add_3P), weights_only=False)
    min, max = train_dataset.normalise()
    
    full_dataset = torch.load("data/full_ic%s_F%s_K%s_3P%s.pt"%(interaction_colors,add_Fatom,add_Katom,add_3P), weights_only=False)
    full_dataset.normalise(min,max) 
    full_loader = DataLoader(full_dataset, batch_size=batch_size_test, shuffle=False)
    err_full,real_dn,out_dn= test(best_model, full_loader,device,criterion,optimizer, minX, maxX, show=True,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom,add_3P=add_3P)
    results['full_mae'].append(err_full.cpu().numpy())
    results['Energy_target_fullSet'].append(real_dn.cpu().numpy())
    results['Energy_predicted_fullSet'].append(out_dn.detach().cpu().numpy())
    
    
    test_dataset = torch.load("data/test_gpu_ic%s_F%s_K%s_%s_3P%s.pt"%(interaction_colors,add_Fatom,add_Katom,ii,add_3P), weights_only=False)
    test_dataset.normalise(min,max) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    err_test,real_dn,out_dn = test(best_model, test_loader,device,criterion,optimizer, minX, maxX, show=True,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom,add_3P=add_3P)
    results['test_mae'].append(err_test.cpu().numpy())
    results['Energy_target_testSet'].append(real_dn.cpu().numpy())
    results['Energy_predicted_testSet'].append(out_dn.detach().cpu().numpy())


df_final = pd.DataFrame(results)
df_final.to_csv('data_Energy_ic%s_F%s_K%s_3P%s.csv'%(interaction_colors,add_Fatom,add_Katom,add_3P))
'''
fig, ax = plt.subplots(figsize=(10,4))

df = pd.read_csv('data_res_ic%s_F%s_K%s_3P%s.csv'%(interaction_colors,add_Fatom,add_Katom,add_3P))

for key, grp in df.groupby('run'):
     ax.plot(grp['epoch'], grp['MAE'], label=key)

ax.text(450, 100, r'MAE(full set)=%0.2f$\pm$(%0.2f)'%(df_final.full_mae.mean(), df_final.full_mae.std(ddof=1)), fontsize=15)
ax.text(450, 150, r'MAE(test set)=%0.2f$\pm$(%0.2f)'%(df_final.test_mae.mean(),df_final.test_mae.std(ddof=1)), fontsize=15)

plt.xlabel('Epoch')
plt.ylabel(f'MAE ($\mu$$E_h$, range: 0 - {int(abs(minX)+abs(maxX))}])')
ax.legend()
plt.title("ic%s_F%s_K%s_3P%s"%(interaction_colors,add_Fatom,add_Katom,add_3P))
plt.savefig("ic%s_F%s_K%s_3P%s_mae.png"%(interaction_colors,add_Fatom,add_Katom,add_3P))
plt.show()
''' 
# ===============================================
