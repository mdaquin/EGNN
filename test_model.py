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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("RUNNIN ON", device)


batch_size_test    = 200 
add_Fatom          = True
add_Katom          = True
interaction_colors = False
learning_rate = 0.001
nRuns     = 10    

df = pd.read_excel("data/data_ia_solol_kmf3.xlsx", skiprows=9, index_col=0).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)
minX = df["dE scaled"].min()
maxX = df["dE scaled"].max()


results = {'run': [], 'full_mae': [], 'test_mae': []}


for ii in range (1,nRuns+1):
    results['run'].append(ii)
    best_model = torch.load('data/best_model_%s_ic%s_F%s_K%s.pt'%(ii,interaction_colors,add_Fatom,add_Katom))
    optimizer = torch.optim.Adam(best_model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss() 
    
    train_dataset = torch.load("data/train_gpu_ic%s_F%s_K%s_%s.pt"%(interaction_colors,add_Fatom,add_Katom,ii), weights_only=False)
    min, max = train_dataset.normalise()
    
    full_dataset = torch.load("data/full_ic%s_F%s_K%s.pt"%(interaction_colors,add_Fatom,add_Katom), weights_only=False)
    full_dataset.normalise(min,max) 
    full_loader = DataLoader(full_dataset, batch_size=batch_size_test, shuffle=False)
    err_full= test(best_model, full_loader,device,criterion,optimizer, minX, maxX, show=True,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom)
    results['full_mae'].append(err_full.cpu().numpy())
    
    
    test_dataset = torch.load("data/test_gpu_ic%s_F%s_K%s_%s.pt"%(interaction_colors,add_Fatom,add_Katom,ii), weights_only=False)
    test_dataset.normalise(min,max) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    err_test = test(best_model, test_loader,device,criterion,optimizer, minX, maxX, show=True,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom)
    results['test_mae'].append(err_test.cpu().numpy())


df_final = pd.DataFrame(results)

