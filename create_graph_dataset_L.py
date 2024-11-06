import sys
import pandas as pd
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils.convert import from_networkx
from egnn.dataset import EGNNDataset 
import random

torch_seed = 42 
torch.manual_seed(torch_seed)
torch.cuda.manual_seed(torch_seed) 
torch.cuda.manual_seed_all(torch_seed) 

def determine_interaction(metal1, metal2, direction):
    """
    Determines the interaction type based on metal colors, orbitals, and direction.

    Parameters:
    metal1 (str): Color of the first metal.
    metal2 (str): Color of the second metal.
    direction (str): Direction of interaction (e.g., "x", "y", "z").

    Returns:
    str: Interaction type (I, K, J, or S).
    """

    color_orbital_map = {
    "r": "yz", 
    "b": "xz", 
    "v": "xy",
    "g": "xy"
    }
    
    orbital1 = color_orbital_map.get(metal1)
    orbital2 = color_orbital_map.get(metal2)
    
    if orbital1 == orbital2:
        if orbital1.__contains__(direction):
            return "I"
        else:
            return "K"
    if orbital1 != orbital2:
        if orbital1.__contains__(direction) and orbital2.__contains__(direction):
            return "J"
        else:
            return "S"
        

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


def graph_from_line_vec(l, G=None, colors=[]):
  colors=[]  
  ng = l.name
  cutoff_distance = 5.0 
  
  if G is None: G=nx.Graph(dE_scaled=l["dE scaled"])
  for i in range(1, 9):
    col = l[f"Color Metal{i}"].lower()
    colR, colG, colB = 1 if col=="r" else 0, 1 if col=="v" else 0, 1 if col=="b" else 0
    G.add_node(f"{ng}_M{i}", colR=colR, colG=colG, colB=colB, atom=l["Z"])

    colors.append(col if col != "v" else "g")
  
  for i in range(1,9):
      for j in range(i+1,9):
               distance=np.round(distMM(i,j,l["a"],l["b"],l["c"]),3)
               if distance <= cutoff_distance:
                   posi = posM(i)
                   posj = posM(j)
                   x = 1 if posi[0] != posj[0] else 0
                   y = 1 if posi[1] != posj[1] else 0
                   z = 1 if posi[2] != posj[2] else 0
                   
                   direction = ""
                   if x == 1:
                       direction += "x"
                   if y == 1:
                       direction += "y"
                   if z == 1:
                       direction += "z"
                   colour1 = l[f"Color Metal{i}"].lower()
                   colour2 = l[f"Color Metal{j}"].lower()
    
                   interaction_type = determine_interaction(colour1, colour2,  direction)
                   
                   interaction_color = {
                "I": "green",
                "K": "grey",
                "J": "red",
                "S": "blue"
            }.get(interaction_type, default_color)
                   
                   
                   colIR, colIGreen, colIB, colIG = 1 if interaction_color=="red" else 0, 1 if interaction_color=="green" else 0, 1 if interaction_color=="blue" else 0, 1 if interaction_color=="grey" else 0
                   #G.add_edge(f"{ng}_M{i}", f"{ng}_M{j}",dx=x, dy=y, dz=z,distance=distance,colIR=colIR, colIGreen=colIGreen, colIB=colIB, colIG=colIG, interaction_color=interaction_color)
                   G.add_edge(f"{ng}_M{i}", f"{ng}_M{j}",dx=x, dy=y, dz=z,distance=distance,colIR=colIR, colIGreen=colIGreen, colIB=colIB, colIG=colIG, interaction_color=interaction_color)

  return G, colors,ng
             
  


def displayGraph(G, ng, colors):
    pos = {f"{ng}_M1": (-0.5, -0.5), f"{ng}_M2": (-0.5, 1), f"{ng}_M3": (1, -0.5), f"{ng}_M4": (1, 1),
           f"{ng}_M5": (-1, -1), f"{ng}_M6": (-1, 0.5), f"{ng}_M7": (0.5, -1), f"{ng}_M8": (0.5, 0.5)}
    plt.figure(figsize=(8,8))
    edge_colors = [G[u][v]['interaction_color'] for u, v in G.edges() if 'interaction_color' in G[u][v]]
    nx.draw(G, with_labels=True, node_size=1000, node_color=colors, pos=pos, edge_color=edge_colors) 
    edge_labels = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, "distance"))
    node_labels = nx.draw_networkx_labels(G, pos=pos)#, labels=nx.get_node_attributes(G, "atom"))
    for u, v, data in G.edges(data=True):
        if 'metal_interaction' in data and data['metal_interaction'] == 1:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red',  width=2) 

    plt.show()



if __name__ == "__main__":
    
    nRand = int(sys.argv[1])
    print("random_state = %s"%(nRand))
#    show_graph = sys.argv[2]
    
    print("*"*6,"loading Data", "*"*6)
    df = pd.read_excel("data/data_ia_solol_kmf3.xlsx", skiprows=9, index_col=0).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)
    print("*"*6,"converting to graphs", "*"*6) 
    df["dE scaled"] = ((df["dE scaled"] - df["dE scaled"].min()) / (df["dE scaled"].max()-df["dE scaled"].min()))
       
    default_color = 'grey'
    
    train_df = df.sample(int(len(df)*0.8), random_state=nRand)
    test_df = df.drop(train_df.index)
    train_list = []
    for l in train_df.iloc: train_list.append(from_networkx(graph_from_line_vec(l)[0]))

    test_list = []
    for l in test_df.iloc: test_list.append(from_networkx(graph_from_line_vec(l)[0]))
    print("*"*6,"saving", "*"*6)
    train = EGNNDataset(train_list)
    test = EGNNDataset(test_list)
    torch.save(train, "data/train_cpu.pt")
    torch.save(test, "data/test_cpu.pt")
    
# =============================================================================
#     if show_graph == True :
#         linenb= random.randint(0,500)
#         G,colors,ng = graph_from_line_vec(train_df.iloc[linenb])
#         displayGraph(G, ng, colors)
#         sys.exit(0)
# =============================================================================

    

