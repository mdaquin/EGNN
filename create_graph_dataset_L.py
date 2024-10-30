import sys
import pandas as pd
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils.convert import from_networkx
from egnn.dataset import EGNNDataset 

def posM(nb):
  match nb:
    case 1: return 0,0,0
    case 2: return 0,0,0.5
    case 3: return 0,0.5,0
    case 4: return 0,0.5,0.5
    case 5: return 0.5,0,0
    case 6: return 0.5,0,0.5
    case 7: return 0.5,0.5,0
    case 8: return 0.5,0.5,0.5

def dist(x1,y1,z1,x2,y2,z2, a, b, c):
  return math.sqrt( (a*(x1-x2))**2 + (b*(y1-y2))**2 + (c*(z1-z2))**2 )

def distMK(nbm, kx, ky, kz, a, b, c):
    met = posM(nbm)
    return dist(met[0], met[1], met[2], kx, ky, kz, a, b, c)

def distMM(nbm1, nbm2, a, b, c):
    met1 = posM(nbm1)
    met2 = posM(nbm2)
    return dist(met1[0], met1[1], met1[2], met2[0], met2[1], met2[2], a, b, c)



def displayGraph(G, ng, colors):
    pos = {f"{ng}_M1": (-0.5, -0.5), f"{ng}_M2": (-0.5, 1), f"{ng}_M3": (1, -0.5), f"{ng}_M4": (1, 1),
           f"{ng}_M5": (-1, -1), f"{ng}_M6": (-1, 0.5), f"{ng}_M7": (0.5, -1), f"{ng}_M8": (0.5, 0.5),
           f"{ng}_F9": (-0.75, -0.75), f"{ng}_F10": (0.25, -0.5), f"{ng}_F11": (-0.5, 0.25),
           f"{ng}_F12": (-0.25, -1), f"{ng}_F13": (-1, -0.25), f"{ng}_F14": (1, 0.25),
           f"{ng}_F15": (0.75, -0.75), f"{ng}_F16": (-0.75, 0.75), f"{ng}_F17": (0.25, 1),
           f"{ng}_F18": (0.5, -0.25), f"{ng}_F19": (-0.25, 0.5), f"{ng}_F20": (0.75, 0.75),
           f"{ng}_K": (0,0)}
    plt.figure(figsize=(8,8))
    nx.draw(G, with_labels=True, node_size=1000, node_color=colors, pos=pos)
    edge_labels = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, "distance"))
    node_labels = nx.draw_networkx_labels(G, pos=pos)#, labels=nx.get_node_attributes(G, "atom"))
    for u, v, data in G.edges(data=True):
        if 'metal_interaction' in data and data['metal_interaction'] == 1:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red',  width=2) 

    plt.show()

def graph_from_line_vec(l, G=None, colors=[]):
  ng = l.name
  if G is None: G=nx.Graph(dE_scaled=l["dE scaled"])
  for i in range(1, 9):
    col = l[f"Color Metal{i}"].lower()
    atom = l["Z"]
    colR, colG, colB = 1 if col=="r" else 0, 1 if col=="v" else 0, 1 if col=="b" else 0
    node_feature = [colR, colG, colB, atom, 1, 0, 0]

    G.add_node(f"{ng}_M{i}", node_feature=node_feature) #, dE_scaled=l["dE scaled"])
    colors.append(col if col != "v" else "g")
  for i in range(9,21):

    node_feature = [0, 0, 0, 9, 0, 1, 0] 
    G.add_node(f"{ng}_F{i}", node_feature=node_feature) 
    colors.append("lightgrey")
  node_feature = [0, 0, 0, 19, 0, 0, 1]  
  G.add_node(f"{ng}_K", node_feature=node_feature)
  colors.append("lightgrey")

  # adding direct links between every metal and every other metal
  # TODO: should there be a link between every F atom as well? and F and the other metals?
  for i in range(1,9):
     for j in range(i,9):
        G.add_edge(f"{ng}_M{i}", f"{ng}_M{j}", distance=np.round(distMM(i,j,l["a"],l["b"],l["c"]), 3), metal_interaction=0) # here is a blue color interaction 
        
        
  G.add_edge(f"{ng}_M6", f"{ng}_M2", metal_interaction=1) # here is a red color interaction 
  G.add_edge(f"{ng}_M8", f"{ng}_M4", metal_interaction=1)
  G.add_edge(f"{ng}_M5", f"{ng}_M1", metal_interaction=1)
  G.add_edge(f"{ng}_M7", f"{ng}_M3", metal_interaction=1)
      
        
  return G, colors,ng

# add on hot encoding for PK and M
# add direct relations between Ms
# try 1,1,1 for color of F and K
# search for neighborhood level in GNN



if __name__ == "__main__":
    print("*"*6,"loading Data", "*"*6)
    df = pd.read_excel("data/data_ia_solol_kmf3.xlsx", skiprows=9, index_col=0).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)
    print("*"*6,"converting to graphs", "*"*6)
    # normalise output
    # TODO: test standardisation ? 
    df["dE scaled"] = ((df["dE scaled"] - df["dE scaled"].min()) / (df["dE scaled"].max()-df["dE scaled"].min()))
    
    linenb= 0
    G,colors,ng = graph_from_line_vec(df.iloc[linenb])
    displayGraph(G, ng, colors)
    
    
    train_df = df.sample(int(len(df)*0.8), random_state=42)
    test_df = df.drop(train_df.index)
    train_list = []
    for l in train_df.iloc: train_list.append(from_networkx(graph_from_line_vec(l)[0]))
    # normalise... 
    test_list = []
    for l in test_df.iloc: test_list.append(from_networkx(graph_from_line_vec(l)[0]))
    print("*"*6,"saving", "*"*6)
    train = EGNNDataset(train_list)
    test = EGNNDataset(test_list)
    torch.save(train, "train_vec.pt")
    torch.save(test, "test_vec.pt")
    

    

