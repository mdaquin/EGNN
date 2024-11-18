import os
from egnn.model_gpu import EGNN
from torch_geometric.loader import DataLoader # type: ignore
import matplotlib.pyplot as plt
import torch
import copy, time
import pandas as pd
from train_gpu import train, test 
import gc

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




learning_rate    = 0.001
batch_size_train = 64
batch_size_test  = 200   

interaction_colors = True 

if interaction_colors == True:
    edge_dimen = 8
else:
    edge_dimen = 4     

input_features = 4 # Default due to 

add_Fatom = True
add_Katom = True

if add_Fatom == True and add_Katom == True:
    input_features = input_features + 3
elif add_Fatom == True and add_Katom == False:
    input_features = input_features + 2 
elif add_Fatom == False and add_Katom == True:
    input_features = input_features + 2 
elif add_Fatom == False and add_Katom == False:
    input_features = input_features  
    

results = {'run': [], 'epoch': [], 'loss': [], 'MAE': []}

nRuns = 10
nepoch = 1000 

for ii in range(1,nRuns+1):
    model = EGNN(input_features=input_features, hidden_channels=256, K=2,edge_dimen = edge_dimen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss() 

    os.system("python3.10 create_graph_dataset_L.py %s "%(ii))
    train_dataset = torch.load("data/train_gpu.pt", weights_only=False)
    min, max = train_dataset.normalise()
    test_dataset = torch.load("data/test_gpu.pt", weights_only=False)
    test_dataset.normalise(min, max)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    
    best_test = None 
    best_epoch = None
    ttt=0
    tte=0
    
    
    
    plt.ion()
    plt.show()
    plt.title('run = %s'%(ii))
    for epoch in range(1, nepoch+1):
        results['run'].append(ii)
        results['epoch'].append(epoch)
        t1 = time.time()
        loss_data = train(model, train_loader,device,criterion,optimizer,interaction_colors=interaction_colors).to(device)
        results['loss'].append(loss_data.detach().cpu().numpy().item())
        tt = round((time.time()-t1)*1000)
        ttt += tt
        t1 = time.time()
        train_acc = test(model, train_loader,device,criterion,optimizer, show=False, clear=True,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom).to(device)
        test_acc = test(model, test_loader,device,optimizer,criterion, show=False,interaction_colors=interaction_colors, add_Fatom =add_Fatom, add_Katom = add_Katom).to(device)
        results['MAE'].append(test_acc.detach().cpu().numpy().item())
        te = round((time.time()-t1)*1000)
        tte += te
        if best_test is None or test_acc < best_test:
            best_test = test_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        print(f'Epoch: {epoch:03d} ({tt:04d}/{te:04d}), Train MAE: {train_acc:.4f}, Test MAE: {test_acc:.4f} (best: {best_test:.4f})')
    
    print("Best MAE on test", best_test,"at",best_epoch)
    print(f"Total time {round(ttt/1000):04d}s for training, {round(tte/1000):04d}s for testing")
    print(f"Average time per epoch {round(ttt/nepoch):04d}ms for training, {round(tte/nepoch):04d}ms for testing")
    test(best_model, test_loader,device,criterion,optimizer, show=True,interaction_colors=interaction_colors)
    
    del model
    torch.cuda.empty_cache()
    plt.ioff()
    plt.show()

df_final = pd.DataFrame(results)
df_final.to_csv('data_res.csv', index=False)
fig, ax = plt.subplots(figsize=(10,4))
plt.title('mae')
plt.xlabel('Epoch')
plt.ylabel('MAE')
for key, grp in df_final.groupby('run'):
    ax.plot(grp['epoch'], grp['MAE'], label=key)

ax.legend()
plt.show()