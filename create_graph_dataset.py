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

def graph_from_line(l, G=None, colors=[]):
  ng = l.name
  # output var is dEscaled
  if G is None: G=nx.Graph(dE_scaled=l["dE scaled"])
  # 8 metal nodes, each with a color and an atom (Z)
  for i in range(1, 9):
    col = l[f"Color Metal{i}"].lower()
    colR, colG, colB = 1 if col=="r" else 0, 1 if col=="v" else 0, 1 if col=="b" else 0
    G.add_node(f"{ng}_M{i}", colR=colR, colG=colG, colB=colB, atom=l["Z"]) #, dE_scaled=l["dE scaled"])
    colors.append(col if col != "v" else "g")
  for i in range(9,21):
    G.add_node(f"{ng}_F{i}", colR=0, colG=0, colB=0, atom=9) #, dE_scaled=l["dE scaled"])
    colors.append("lightgrey")
  G.add_node(f"{ng}_K", colR=0, colG=0, colB=0, atom=19) #, dE_scaled=l["dE scaled"])
  colors.append("lightgrey")
  # Each M has a shift wrt a F atom which is a factor of a b or c depending on the direction
  G.add_edge(f"{ng}_M1", f"{ng}_F9" , distance=np.round(distMK(1,0.25+l["M1 shift xF9" ]*0.0001,0,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F9", f"{ng}_M5" , distance=np.round(distMK(5,0.25+l["M1 shift xF9" ]*0.0001,0,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M4", f"{ng}_F20", distance=np.round(distMK(4,0.25+l["M4 shift xF20"]*0.0001,0.5,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F20", f"{ng}_M8", distance=np.round(distMK(8,0.25+l["M4 shift xF20"]*0.0001,0.5,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M6", f"{ng}_F16", distance=np.round(distMK(6,0.25+l["M6 shift xF16"]*0.0001,0,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F16", f"{ng}_M2", distance=np.round(distMK(2,0.25+l["M6 shift xF16"]*0.0001,0,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M7", f"{ng}_F15", distance=np.round(distMK(7,0.25+l["M7 shift xF15"]*0.0001,0.5,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F15", f"{ng}_M3", distance=np.round(distMK(3,0.25+l["M7 shift xF15"]*0.0001,0.5,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M1", f"{ng}_F10", distance=np.round(distMK(1,0,0.25+l["M1 shift yF10"]*0.0001,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F10", f"{ng}_M3", distance=np.round(distMK(3,0,0.25+l["M1 shift yF10"]*0.0001,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M4", f"{ng}_F17", distance=np.round(distMK(4,0,0.25+l["M4 shift yF17"]*0.0001,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F17", f"{ng}_M2", distance=np.round(distMK(2,0,0.25+l["M4 shift yF17"]*0.0001,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M6", f"{ng}_F19", distance=np.round(distMK(6,0.5,0.25+l["M6 shift yF19"]*0.0001,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F19", f"{ng}_M8", distance=np.round(distMK(8,0.5,0.25+l["M6 shift yF19"]*0.0001,0.5,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M7", f"{ng}_F12", distance=np.round(distMK(7,0.5,0.25+l["M7 shift yF12"]*0.0001,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F12", f"{ng}_M5", distance=np.round(distMK(5,0.5,0.25+l["M7 shift yF12"]*0.0001,0,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M1", f"{ng}_F11", distance=np.round(distMK(1,0,0,0.25+l["M1 shift zF11"]*0.0001,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F11", f"{ng}_M2", distance=np.round(distMK(2,0,0,0.25+l["M1 shift zF11"]*0.0001,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M4", f"{ng}_F14", distance=np.round(distMK(4,0,0.5,0.25+l["M4 shift zF14"]*0.0001,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F14", f"{ng}_M3", distance=np.round(distMK(3,0,0.5,0.25+l["M4 shift zF14"]*0.0001,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_M6", f"{ng}_F13", distance=np.round(distMK(6,0.5,0,0.25+l["M6 shift zF13"]*0.0001,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F13", f"{ng}_M5", distance=np.round(distMK(5,0.5,0,0.25+l["M6 shift zF13"]*0.0001,l["a"],l["b"],l["c"]),3))
  # here the column is called M6 shift zF18, but it should be M8 shift zF18...
  G.add_edge(f"{ng}_M8", f"{ng}_F18", distance=np.round(distMK(8,0.5,0.5,0.25+l["M6 shift zF18"]*0.0001,l["a"],l["b"],l["c"]),3))
  G.add_edge(f"{ng}_F18", f"{ng}_M7", distance=np.round(distMK(7,0.5,0.5,0.25+l["M6 shift zF18"]*0.0001,l["a"],l["b"],l["c"]),3))
  # the K atom has a 3 directional shift as a factor of a b or c
  kx,ky,kz = (0.25+(l["K shift x"]*0.001)),(0.25+(l["K shift y"]*0.001)),(0.25+(l["K shift z"]*0.001))
  for i in range(1,9):
    G.add_edge(f"{ng}_M{i}", f"{ng}_K", distance=np.round(distMK(i, kx, ky, kz, l["a"], l["b"], l["c"]),3))
  G.add_edge(f"{ng}_F9", f"{ng}_K", distance=np.round(dist(0.25+l["M1 shift xF9" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F20", f"{ng}_K", distance=np.round(dist(0.25+l["M4 shift xF20" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F16", f"{ng}_K", distance=np.round(dist(0.25+l["M6 shift xF16" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F15", f"{ng}_K", distance=np.round(dist(0.25+l["M7 shift xF15" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F10", f"{ng}_K", distance=np.round(dist(0,0.25+l["M1 shift yF10" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F17", f"{ng}_K", distance=np.round(dist(0,0.25+l["M4 shift yF17" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F19", f"{ng}_K", distance=np.round(dist(0,0.25+l["M6 shift yF19" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F12", f"{ng}_K", distance=np.round(dist(0,0.25+l["M7 shift yF12" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F11", f"{ng}_K", distance=np.round(dist(0,0,0.25+l["M1 shift zF11" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F14", f"{ng}_K", distance=np.round(dist(0,0,0.25+l["M4 shift zF14" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F13", f"{ng}_K", distance=np.round(dist(0,0,0.25+l["M6 shift zF13" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  G.add_edge(f"{ng}_F18", f"{ng}_K", distance=np.round(dist(0,0,0.25+l["M6 shift zF18" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3))
  return G, colors

def displayGraph(G, ng, colors):
    pos = {f"{ng}_M1": (-0.5, -0.5), f"{ng}_M2": (-0.5, 1), f"{ng}_M3": (1, -0.5), f"{ng}_M4": (1, 1),
           f"{ng}_M5": (-1, -1), f"{ng}_M6": (-1, 0.5), f"{ng}_M7": (0.5, -1), f"{ng}_M8": (0.5, 0.5),
           f"{ng}_F9": (-0.75, -0.75), f"{ng}_F10": (0.25, -0.5), f"{ng}_F11": (-0.5, 0.25),
           f"{ng}_F12": (-0.25, -1), f"{ng}_F13": (-1, -0.25), f"{ng}_F14": (1, 0.25),
           f"{ng}_F15": (0.75, -0.75), f"{ng}_F16": (-0.75, 0.75), f"{ng}_F17": (0.25, 1),
           f"{ng}_F18": (0.5, -0.25), f"{ng}_F19": (-0.25, 0.5), f"{ng}_F20": (0.75, 0.75),
           f"{ng}_K": (0,0)}
    plt.figure(figsize=(8,8))
    # should create positions myself...
    nx.draw(G, with_labels=True, node_size=1000, node_color=colors, pos=pos)
    edge_labels = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, "distance"))
    node_labels = nx.draw_networkx_labels(G, pos=pos)#, labels=nx.get_node_attributes(G, "atom"))
    plt.show()


def displayLargeGraph(G, colors):
    plt.figure(figsize=(10,10))
    # should create positions myppself...
    nx.draw(G, with_labels=True, node_size=1000, node_color=colors)
    # edge_labels = nx.draw_networkx_edge_labels(G, edge_labels=nx.get_edge_attributes(G, "distance"))
    # node_labels = nx.draw_networkx_labels(G)#, labels=nx.get_node_attributes(G, "atom"))
    plt.show()

if __name__ == "__main__":
    print("*"*6,"loading Data", "*"*6)
    df = pd.read_excel("data/data_ia_solol_kmf3.xlsx", skiprows=9, index_col=0).drop(["Nb V", "Nb B", "Nb R", "Label"], axis=1)
    print("*"*6,"converting to graphs", "*"*6)
    train_df = df.sample(int(len(df)*0.8), random_state=42)
    test_df = df.drop(train_df.index)
    train_list = []
    for l in train_df.iloc: train_list.append(from_networkx(graph_from_line(l)[0]))
    test_list = []
    for l in test_df.iloc: test_list.append(from_networkx(graph_from_line(l)[0]))
    print("*"*6,"saving", "*"*6)
    train = EGNNDataset(train_list)
    test = EGNNDataset(test_list)
    torch.save(train, "data/train.pt")
    torch.save(test, "data/test.pt")

    

