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

torch.cuda.empty_cache()
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


def add_fluorine_metaal_connection (G,l,default_colorFM='black'):
    G.add_edge(f"{ng}_M1", f"{ng}_F9" ,
           dx=1, dy=0, dz=0,
           distance=np.round(distMK(1,0.25+l["M1 shift xF9" ]*0.0001,0,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F9", f"{ng}_M5" ,
               dx=1, dy=0, dz=0,
               distance=np.round(distMK(5,0.25+l["M1 shift xF9" ]*0.0001,0,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M4", f"{ng}_F20",
               dx=1, dy=0, dz=0,
               distance=np.round(distMK(4,0.25+l["M4 shift xF20"]*0.0001,0.5,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F20", f"{ng}_M8",
               dx=1, dy=0, dz=0,
               distance=np.round(distMK(8,0.25+l["M4 shift xF20"]*0.0001,0.5,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M6", f"{ng}_F16",
               dx=1, dy=0, dz=0,
               distance=np.round(distMK(6,0.25+l["M6 shift xF16"]*0.0001,0,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F16", f"{ng}_M2",
               dx=1, dy=0, dz=0,
               distance=np.round(distMK(2,0.25+l["M6 shift xF16"]*0.0001,0,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M7", f"{ng}_F15",
               dx=1, dy=0, dz=0,
               distance=np.round(distMK(7,0.25+l["M7 shift xF15"]*0.0001,0.5,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F15", f"{ng}_M3",
               dx=1, dy=0, dz=0,
               distance=np.round(distMK(3,0.25+l["M7 shift xF15"]*0.0001,0.5,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M1", f"{ng}_F10",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(1,0,0.25+l["M1 shift yF10"]*0.0001,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F10", f"{ng}_M3",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(3,0,0.25+l["M1 shift yF10"]*0.0001,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M4", f"{ng}_F17",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(4,0,0.25+l["M4 shift yF17"]*0.0001,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F17", f"{ng}_M2",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(2,0,0.25+l["M4 shift yF17"]*0.0001,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M6", f"{ng}_F19",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(6,0.5,0.25+l["M6 shift yF19"]*0.0001,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F19", f"{ng}_M8",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(8,0.5,0.25+l["M6 shift yF19"]*0.0001,0.5,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M7", f"{ng}_F12",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(7,0.5,0.25+l["M7 shift yF12"]*0.0001,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F12", f"{ng}_M5",
               dx=0, dy=1, dz=0,
               distance=np.round(distMK(5,0.5,0.25+l["M7 shift yF12"]*0.0001,0,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M1", f"{ng}_F11",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(1,0,0,0.25+l["M1 shift zF11"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F11", f"{ng}_M2",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(2,0,0,0.25+l["M1 shift zF11"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M4", f"{ng}_F14",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(4,0,0.5,0.25+l["M4 shift zF14"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F14", f"{ng}_M3",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(3,0,0.5,0.25+l["M4 shift zF14"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_M6", f"{ng}_F13",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(6,0.5,0,0.25+l["M6 shift zF13"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F13", f"{ng}_M5",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(5,0.5,0,0.25+l["M6 shift zF13"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    # here the column is called M6 shift zF18, but it should be M8 shift zF18...
    G.add_edge(f"{ng}_M8", f"{ng}_F18",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(8,0.5,0.5,0.25+l["M6 shift zF18"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)
    G.add_edge(f"{ng}_F18", f"{ng}_M7",
               dx=0, dy=0, dz=1,
               distance=np.round(distMK(7,0.5,0.5,0.25+l["M6 shift zF18"]*0.0001,l["a"],l["b"],l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFM)


def add_fluorine_potasium_connection(G,l,default_colorFK='black') :
      kx,ky,kz = (0.25+(l["K shift x"]*0.001)),(0.25+(l["K shift y"]*0.001)),(0.25+(l["K shift z"]*0.001))
      G.add_edge(f"{ng}_F9", f"{ng}_K",
             dx=0, dy=0, dz=0,
             distance=np.round(dist(0.25+l["M1 shift xF9" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F20", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0.25+l["M4 shift xF20" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F16", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0.25+l["M6 shift xF16" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F15", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0.25+l["M7 shift xF15" ]*0.0001,0,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F10", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0.25+l["M1 shift yF10" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F17", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0.25+l["M4 shift yF17" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F19", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0.25+l["M6 shift yF19" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F12", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0.25+l["M7 shift yF12" ]*0.0001,0,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F11", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0,0.25+l["M1 shift zF11" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F14", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0,0.25+l["M4 shift zF14" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F13", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0,0.25+l["M6 shift zF13" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)
      G.add_edge(f"{ng}_F18", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(dist(0,0,0.25+l["M6 shift zF18" ]*0.0001,kx,ky,kz,l["a"],l["b"],l["c"]), 3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_colorFK)

    

def graph_from_line_vec(l,default_color='black',add_Katom = False, add_Fatom = False, G=None, colors=[]):
  colors=[]  
  ng = l.name
  cutoff_distance = 5.0 
  
  if G is None: G=nx.Graph(dE_scaled=l["dE scaled"])
  for i in range(1, 9):
      col = l[f"Color Metal{i}"].lower()
      colR, colG, colB = 1 if col=="r" else 0, 1 if col=="v" else 0, 1 if col=="b" else 0
      G.add_node(f"{ng}_M{i}", colR=colR, colG=colG, colB=colB, atom=l["Z"], metal=1, fluoride=0, potassium=0) 
      colors.append(col if col != "v" else "g")
      
  if add_Katom == False:
      if add_Fatom == False:
          pass   
      elif add_Fatom == True:               
          for i in range(9,21):
                G.add_node(f"{ng}_F{i}", colR=0, colG=0, colB=0, atom=9, metal=0, fluoride=1,colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_color) 
                colors.append("lightgrey")
          add_fluorine_metaal_connection(G,l)
          
  else:
      kx,ky,kz = (0.25+(l["K shift x"]*0.001)),(0.25+(l["K shift y"]*0.001)),(0.25+(l["K shift z"]*0.001))
      if add_Fatom == False:
          G.add_node(f"{ng}_K", colR=0, colG=0, colB=0, atom=19, metal=0, potassium=1)
          colors.append("lightgrey")
          for i in range(1,9):
              G.add_edge(f"{ng}_M{i}", f"{ng}_K",
                 dx=0, dy=0, dz=0,
                 distance=np.round(distMK(i, kx, ky, kz, l["a"], l["b"], l["c"]),3),colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_color)
      else:
        for i in range(9,21):
            G.add_node(f"{ng}_F{i}", colR=0, colG=0, colB=0, atom=9, metal=0, fluoride=1, potassium=0,colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_color) 
            colors.append("lightgrey")
        G.add_node(f"{ng}_K", colR=0, colG=0, colB=0, atom=19, metal=0, fluoride=0, potassium=1,colIR=0, colIGreen=0, colIB=0, colIG=0, interaction_color=default_color) 
        colors.append("lightgrey")
        add_fluorine_metaal_connection(G,l)
        add_fluorine_potasium_connection(G,l)
        
          
  
    
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
            }.get(interaction_type, 'None')
                   
                   
                   colIR, colIGreen, colIB, colIG = 1 if interaction_color=="red" else 0, 1 if interaction_color=="green" else 0, 1 if interaction_color=="blue" else 0, 1 if interaction_color=="grey" else 0
                   #G.add_edge(f"{ng}_M{i}", f"{ng}_M{j}",dx=x, dy=y, dz=z,distance=distance,colIR=colIR, colIGreen=colIGreen, colIB=colIB, colIG=colIG, interaction_color=interaction_color)
                   G.add_edge(f"{ng}_M{i}", f"{ng}_M{j}",dx=x, dy=y, dz=z,distance=distance,colIR=colIR, colIGreen=colIGreen, colIB=colIB, colIG=colIG, interaction_color=interaction_color)

  return G, colors,ng
             
  


def displayGraph2(G, ng, colors):
    pos = {f"{ng}_M1": (-0.5, -0.5), f"{ng}_M2": (-0.5, 1), f"{ng}_M3": (1, -0.5), f"{ng}_M4": (1, 1),
           f"{ng}_M5": (-1, -1), f"{ng}_M6": (-1, 0.5), f"{ng}_M7": (0.5, -1), f"{ng}_M8": (0.5, 0.5),
           f"{ng}_F9": (-0.75, -0.75), f"{ng}_F10": (0.25, -0.5), f"{ng}_F11": (-0.5, 0.25),
           f"{ng}_F12": (-0.25, -1), f"{ng}_F13": (-1, -0.25), f"{ng}_F14": (1, 0.25),
           f"{ng}_F15": (0.75, -0.75), f"{ng}_F16": (-0.75, 0.75), f"{ng}_F17": (0.25, 1),
           f"{ng}_F18": (0.5, -0.25), f"{ng}_F19": (-0.25, 0.5), f"{ng}_F20": (0.75, 0.75),
           f"{ng}_K": (0,0)}
    plt.figure(figsize=(8,8))
     
    edge_colors = [G.edges[u, v]['interaction_color'] for u, v in G.edges()]
    nx.draw(G, with_labels=True, node_size=1000, node_color=colors, pos=pos,edge_color=edge_colors)

    #nx.draw(G, with_labels=True, node_size=1000, node_color=colors, pos=pos) # Use node_colors instead of colors
    edge_labels = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, "distance"))
    node_labels = nx.draw_networkx_labels(G, pos=pos)#, labels=nx.get_node_attributes(G, "atom"))
    plt.show()


def displayGraph(G, ng, colors):
    pos = {f"{ng}_M1": (-0.5, -0.5), f"{ng}_M2": (-0.5, 1), f"{ng}_M3": (1, -0.5), f"{ng}_M4": (1, 1),
           f"{ng}_M5": (-1, -1), f"{ng}_M6": (-1, 0.5), f"{ng}_M7": (0.5, -1), f"{ng}_M8": (0.5, 0.5),
           f"{ng}_F9": (-0.75, -0.75), f"{ng}_F10": (0.25, -0.5), f"{ng}_F11": (-0.5, 0.25),
           f"{ng}_F12": (-0.25, -1), f"{ng}_F13": (-1, -0.25), f"{ng}_F14": (1, 0.25),
           f"{ng}_F15": (0.75, -0.75), f"{ng}_F16": (-0.75, 0.75), f"{ng}_F17": (0.25, 1),
           f"{ng}_F18": (0.5, -0.25), f"{ng}_F19": (-0.25, 0.5), f"{ng}_F20": (0.75, 0.75),
           f"{ng}_K": (0,0)}
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
    torch.save(train, "data/train_gpu.pt")
    torch.save(test, "data/test_gpu.pt")
    
# =============================================================================
#     if show_graph == True :
#         linenb= random.randint(0,500)
#         G,colors,ng = graph_from_line_vec(train_df.iloc[linenb])
#         displayGraph(G, ng, colors)
#         sys.exit(0)
# =============================================================================

    

