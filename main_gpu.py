import os
from egnn.model_gpu import EGNN
from torch_geometric.loader import DataLoader # type: ignore
import matplotlib.pyplot as plt
import torch
import copy, time
import pandas as pd
from train_gpu import train, test, sizeofmodel 
import gc
import json,sys
from create_graph_dataset import create_graph


def create_dataset (): 
    scaling_factor =  {'sc':1,'ti':1.498, 'fe':2.187, 'co':3.075 } 

    compound_name = 'sc' 
    df_1 = pd.read_csv("../GEGNN/data/table_ia_%s_b3.csv"%(compound_name)).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)     
    compound_name = 'ti' 
    df_2 = pd.read_csv("../GEGNN/data/table_ia_%s_b3.csv"%(compound_name)).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)   
    compound_name = 'fe' 
    df_3 = pd.read_csv("../GEGNN/data/table_ia_%s_b3.csv"%(compound_name)).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)  
    compound_name = 'co' 
    df_4 = pd.read_csv("../GEGNN/data/table_ia_%s_b3.csv"%(compound_name)).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)

    mask = 1000 
    
    df = pd.concat([df_4[:mask], df_2[:mask],df_3[:mask],df_1[:mask]], sort=False, ignore_index=True)
    df.to_csv('data/all_data.csv')
    
    return df 


# =============================================================================
# if len(sys.argv) != 2:
#    print("please provide a config file")
#    sys.exit(-1)
# 
# =============================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
gc.collect()

torch_seed = 42 
torch.cuda.empty_cache()
torch.manual_seed(torch_seed)
torch.cuda.manual_seed(torch_seed) 
torch.cuda.manual_seed_all(torch_seed) 

# =============================================================================
#  
#   Setting the parameters for the train / test of the model 
# 
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("RUNNIN ON", device)


with open(sys.argv[1]) as f:
   params = json.load(f)

#with open("input_config_FFF.json") as f:
#   params = json.load(f)

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
# nRuns = 1 
# add_3P = False
# add_Fatom = False
# add_Katom = False 
# interaction_colors=False
# =============================================================================


edge_dimen,input_features = sizeofmodel (add_Fatom,add_Katom,interaction_colors,add_3P)
results = {'run': [], 'epoch': [], 'loss': [], 'MAE': []}
#df = pd.read_excel("data/data_ia_solol_kmf3.xlsx", skiprows=9, index_col=0).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)
df = pd.read_csv('data/all_new_data_corrected.csv')


minX = df["dE scaled"].min()
maxX = df["dE scaled"].max()


for ii in range(1,nRuns+1):
    model = EGNN(input_features=input_features, hidden_channels=hidden_channels, K=2,edge_dimen = edge_dimen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss() 

    #os.system("python3.10 create_graph_dataset.py %s %s %s %s"%(ii,add_Fatom,add_Katom,interaction_colors)) 
    
    create_graph(ii,add_Fatom,add_Katom,add_3P,interaction_colors)
    
    train_dataset = torch.load("data/train_gpu_ic%s_F%s_K%s_%s_3P%s.pt"%(interaction_colors,add_Fatom,add_Katom,ii,add_3P), weights_only=False)
    min, max = train_dataset.normalise()
    test_dataset = torch.load("data/test_gpu_ic%s_F%s_K%s_%s_3P%s.pt"%(interaction_colors,add_Fatom,add_Katom,ii,add_3P), weights_only=False)
    test_dataset.normalise(min, max)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    
    best_test = None 
    best_epoch = None
    ttt=0
    tte=0
    
    
    
    #plt.ion()
    #plt.show()
    #plt.title('run = %s'%(ii))
    for epoch in range(1, nepoch+1):
        results['run'].append(ii)
        results['epoch'].append(epoch)
        t1 = time.time()
        loss_data = train(model, train_loader,device,criterion,optimizer, minX, maxX,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom,add_3P=add_3P).to(device)
        results['loss'].append(loss_data.detach().cpu().numpy().item())
        tt = round((time.time()-t1)*1000)
        ttt += tt
        t1 = time.time()
        train_acc = test(model, train_loader,device,criterion,optimizer, minX, maxX, show=False, clear=True,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom,add_3P=add_3P).to(device)
        test_acc = test(model, test_loader,device,criterion, optimizer, minX, maxX, show=False,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom,add_3P=add_3P).to(device)
        results['MAE'].append(test_acc.detach().cpu().numpy().item())
        te = round((time.time()-t1)*1000)
        tte += te
        if best_test is None or test_acc < best_test:
            best_test = test_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            torch.save(best_model.state_dict(), 'data/state_best_model_%s_ic%s_F%s_K%s_3P%s.pt' % (ii, interaction_colors, add_Fatom, add_Katom, add_3P))
            torch.save(best_model,'data/best_model_%s_ic%s_F%s_K%s_3P%s.pt'%(ii,interaction_colors,add_Fatom,add_Katom,add_3P))
        print(f'Epoch: {epoch:03d} ({tt:04d}/{te:04d}), Train MAE: {train_acc:.4f}, Test MAE: {test_acc:.4f} (best: {best_test:.4f})')
    
    print("Best MAE on test", best_test,"at",best_epoch)
    print(f"Total time {round(ttt/1000):04d}s for training, {round(tte/1000):04d}s for testing")
    print(f"Average time per epoch {round(ttt/nepoch):04d}ms for training, {round(tte/nepoch):04d}ms for testing")
    test(best_model, test_loader,device,criterion,optimizer, minX, maxX, show=False,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom,add_3P=add_3P)
    
    del model
    torch.cuda.empty_cache()
    #plt.ioff()
    #plt.show()

df_final = pd.DataFrame(results)
df_final.to_csv('data_res_ic%s_F%s_K%s_3P%s.csv'%(interaction_colors,add_Fatom,add_Katom,add_3P), index=False)
#fig, ax = plt.subplots(figsize=(10,4))

# =============================================================================
# df_final = pd.read_csv('data_res_icFalse_FFalse_KFalse.csv')
# lowest_mae_per_run = df_final.loc[df_final.groupby('run')['MAE'].idxmin()]['MAE']
# last_mae = df_final[df_final['epoch'] == 1000]['MAE']
# fig, ax = plt.subplots(figsize=(10,4))
# ax.text(450, 500, r'MAE(last epoch)=%0.2f$\pm$(%0.2f)'%(last_mae.mean(), last_mae.std(ddof=1)), fontsize=15)
# ax.text(450, 700, r'MAE(lowest)=%0.2f$\pm$(%0.2f)'%(lowest_mae_per_run.mean(),lowest_mae_per_run.std(ddof=1)), fontsize=15)
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# for key, grp in df_final.groupby('run'):
#     ax.plot(grp['epoch'], grp['MAE'], label=key)
# 
# ax.legend()
# 
# plt.show()
# =============================================================================
