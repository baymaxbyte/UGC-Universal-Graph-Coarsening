from locale import currency
import math
from pickle import FALSE
from re import L
from unicodedata import name
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric
try:
    from scatter_letters import sl  # only needed for --scatter_alphabets mode
except Exception:
    sl = None

import seaborn as sns
from sklearn.manifold import TSNE

from scipy.spatial.distance import cdist

from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Yelp
from torch_geometric.datasets import AmazonProducts
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import AMiner
from torch_geometric.datasets import OGB_MAG
from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import scipy.io

import os
import json
import scipy as sp
from scipy.sparse import csr_matrix
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
#import tensorflow as tf
import argparse
import time
from scipy.spatial.distance import pdist
from itertools import chain

import pygsp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Compatibility shim for PyTorch >= 2.6 ---
# This repo saves whole models via torch.save(model) and reloads them with
# torch.load(...). PyTorch 2.6 changed torch.load's default weights_only to
# True, which cannot unpickle full model objects. Restore legacy behavior.
if not getattr(torch, "_ugc_load_patched", False):
    _orig_torch_load = torch.load
    def _compat_torch_load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_torch_load(*a, **k)
    torch.load = _compat_torch_load
    torch._ugc_load_patched = True

import utils
import GCN
import spectral_properties
import UGC_bin_widths
import GraphSage
import GAT
import GIN
import APPNP_GCN as APPNP
import WL_base_model

def parse_args():
    parser = argparse.ArgumentParser(description='Coarsened Graph Training')
    parser.add_argument('--full_dataset',type=bool,required=False,default=False,help="Checking accuracy on original dataset.")
    parser.add_argument('--dataset',type=str,required=False,default='cora',help="Dataset name")
    parser.add_argument('--edge_index_path',type=str,required=False,default='None',help="Give path of edge index file")
    parser.add_argument('--label_path',type=str,required=False,default='None',help="Give path of label file")
    parser.add_argument('--node_feat_path',type=str,required=False,default='None',help="Give path of node feature file")
    parser.add_argument('--add_adj_to_node_features',type=bool,required=False,default=True,help="Adding Adjacency matrix one hot vectors in node features")
    parser.add_argument('--epochs',type=int,required=False, default=500,help="Number of epochs to train the coarsened graph")
    parser.add_argument('--lr',type=float,required=False,default=0.003,help="Learning Rate")
    parser.add_argument('--decay',type=float,required=False,default=0.0005,help="Learning Rate Decay")
    parser.add_argument('--seed',type=int,required=False,default=42,help="Seed")
    parser.add_argument('--ratio',type=int,required=False,default=50,help='reduction ratio list, example (30,50,70)')
    parser.add_argument('--dataset_not_in_torch_geometric',type=bool,required=False,default=False,help='Turn true if your dataset is not in the torch geometric. We will create geometric dataset first')
    parser.add_argument('--num_classes',type=int,required=False,default=-1,help='You should give value here if new instance of torch_geometric dataset is being created.')
    parser.add_argument('--number_of_projectors',type=int,required=False,default=500,help='Total number of projectors we want while Doing LSH.')
    parser.add_argument('--out_of_sample',type=int,required=False,default=0,help='UGC2.0 should be supporting this. out_of_sample in percent (from 0 to 1) of dataset')
    parser.add_argument('--feature_size',type=int,required=False,default=-1,help='You should give value here if new instance of torch_geometric dataset is being created.')
    parser.add_argument('--hash_function',type=str,required=False,default='dot',help='Hash Function choices 1). Dot 2). L1-norm 3). L2-norm')
    parser.add_argument('--projectors_distribution',type=str,required=False,default='uniform',help='1). uniform 2). normal. coming soon.... 3). VAEs in this case need to give learned mean and sigma also.')
    parser.add_argument('--random_coarsening',type=bool,required=False,default=False,help='True for random coarsening.')
    parser.add_argument('--visualize_graph',type=bool,required=False,default=False,help='True for graph visualization.')
    parser.add_argument('--induce_adverserial_edges',type=bool,required=False,default=False,help='True for adding noise in the graph edges.')
    parser.add_argument('--tsne_visualization',type=bool,required=False,default=False,help='tsne_visualization')
    parser.add_argument('--calculate_spectral_errors',type=bool,required=False,default=False,help='calculate_spectral_errors')
    parser.add_argument('--hidden_units',type=int,required=False,default=64,help='hidden_units of GCN')
    parser.add_argument('--gsp_graphs',type=bool,required=False,default=False,help='making graphs from Graph Signal Processing lib')
    parser.add_argument('--scatter_alphabets',type=str,required=False,default="None",help='making graphs from names and alphabets')
    parser.add_argument('--alpha',type=float,required=False,default=0.1,help='Heterophilic factor')
    parser.add_argument('--model_type',type=str,required=False,default='gcn',help='model type')
    parser.add_argument('--save_coarsened',type=bool,required=False,default=False,help='Save the coarsened graph and partition matrix to coarsened_data_UGC/')
    
    args = parser.parse_args()
    return args

def hashed_values(data, no_of_hash,feature_size,function,out_of_sample,projectors_distribution,A):

  if projectors_distribution == 'VAEs':
    print("some random intilization is given here for mean and sigma make sure these contain learned values")
    learned_mean = -0.0017
    learned_sigma = 0.29
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(learned_mean,learned_sigma)
    # Wl = torch.FloatTensor(vecs)
  elif projectors_distribution == 'karate':
     Wl = [] #torch.FloatTensor(utils.sample_projectors(vecs,no_of_hash,feature_size))
  elif projectors_distribution == 'normal':
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0,1)
  else:
    #uniform
    Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0,1)
  
  if out_of_sample != 0:
    num_out_of_sample = (int)(data.num_nodes*(1 - out_of_sample))
    idx = np.random.randint(data.num_nodes, size=num_out_of_sample)
    out_of_sampled_data_x = data.x[idx,:]
  else:
    out_of_sampled_data_x = data.x

  if function == 'L2-norm':
    Bin_values = torch.cdist(out_of_sampled_data_x, Wl, p = 2)
  elif function == 'L1-norm':
    Bin_values = torch.cdist(out_of_sampled_data_x, Wl, p = 1)
  else:
    #dot
    Bin_values = torch.matmul(out_of_sampled_data_x, Wl.T)
  
  return Bin_values

def allocate_list_bin_width(dataset_name,ratio_list,hash_function,scatter_alphabets):
  if scatter_alphabets == 'None':
    key = dataset_name + '_' + hash_function
    full_bin_width_list =  UGC_bin_widths.BIN_WIDTH_DICTONARY[key] 
    list_bin_width = []
    for ratio in ratio_list:
      key = (str)(ratio)
      list_bin_width.append(full_bin_width_list[key]) 
  else:
     list_bin_width = [0.3]
  # print(list_bin_width)
  return list_bin_width

def partition(list_bin_width,Bin_values,no_of_hash):
    summary_dict = {}
    print(list_bin_width)
    for bin_width in list_bin_width:
        bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)])#.to(device)
        temp = torch.floor((1/bin_width)*(Bin_values + bias))#.to(device)

        cluster, _ = torch.mode(temp, dim = 1)
        dict_hash_indices = {}
        no_nodes = Bin_values.shape[0]
        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i]) #.to('cpu')
        summary_dict[bin_width] = dict_hash_indices 

        # min_value = torch.min(temp)
        # max_value = torch.max(temp)
        # chunks = np.arange(min_value, max_value + bin_width, bin_width)  #Creating bins with size r
        # chunk_indices = np.digitize(temp, chunks) - 1  
        # dict_hash_indices = {}
        # i = 0
        # for row in chunk_indices:
        #     counts = Counter(row)
        #     most_common_chunk = counts.most_common(1)[0][0]
        #     dict_hash_indices[i] = most_common_chunk
        #     i = i + 1
        
        # summary_dict[bin_width] = dict_hash_indices 
    return summary_dict

def val(model,data):
    data = data#.to(device)
    model.eval()
    if args.model_type not in ['gcn','3wl']:
      pred = model(data.x, data.edge_index).argmax(dim=1)
    elif args.model_type == '3wl':
      pred = model(data.x).argmax(1)
    else:
      pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
    
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc

def split(data, num_classes,split_percent):
    indices = []
    num_test = (int)(data.num_nodes * split_percent / num_classes)
    for i in range(num_classes):
        index = (data.y == i).nonzero().reshape(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    
    test_index = torch.cat([i[:num_test] for i in indices], dim=0)
    val_index = torch.cat([i[num_test:int(num_test*1.5)] for i in indices], dim=0)
    train_index = torch.cat([i[int(num_test*1.5):] for i in indices], dim=0)

    # print(train_index)

    data.train_mask = utils.index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = utils.index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = utils.index_to_mask(test_index, size=data.num_nodes)
    return data


def train_on_original_dataset(data, num_classes, feature_size, hidden_units, learning_rate, decay, epochs):
  # model = WL_base_model.WL_BaseModel(feature_size, hidden_units, num_classes)
  model = GCN.GCN_(feature_size, hidden_units, num_classes)
  # model = GraphSage.GraphSAGE(feature_size, hidden_units, num_classes)
  # model = GAT.GAT(feature_size, hidden_units, num_classes)
  # model = GIN.GIN(feature_size, hidden_units, num_classes)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)
  test_split_percent = 0.2
  data = split(data,num_classes,test_split_percent)
  
  if data.edge_attr == None:
    edge_weight = torch.ones(data.edge_index.size(1))
    data.edge_attr = edge_weight
    
  for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index,data.edge_attr.float())
    # out = model(data.x, data.edge_index)
    # out = model(data.x)
    pred = out.argmax(1)
    criterion = torch.nn.NLLLoss()
    
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    best_val_acc = 0
    
    val_acc = val(model,data)
    if best_val_acc < val_acc:
        torch.save(model, 'full_best_model.pt')
        best_val_acc = val_acc
  
    if epoch % 100 == 0:
        print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))

  model = torch.load('full_best_model.pt')
  model.eval()
  data = data#.to(device)
  pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
  # pred = model(data.x, data.edge_index).argmax(dim=1)
  # pred = model(data.x).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())

  incorrect_indices = (pred[data.test_mask] != data.y[data.test_mask]).nonzero()

  # Convert the indices to a list
  incorrect_indices_list = incorrect_indices.view(-1).tolist()
    
  print('--------------------------')
  print('Accuracy on test data {:.3f}'.format(acc*100))

  return incorrect_indices_list


#################
def handling_gsp_graphs(G, with_labels, scatter_alphabets):
  # print(G.W.shape)
  # print(G.labels)
  #print(G.W)
  if scatter_alphabets == 'None':
    adj_matrix = G.W.toarray()
  else:
     adj_matrix = G.W
  
  # node_degrees = np.sum(adj_matrix, axis=1)  
  # node_features = (node_degrees - np.min(node_degrees)) / (np.max(node_degrees) - np.min(node_degrees))


  edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
  #x = torch.tensor(node_features, dtype=torch.float).unsqueeze(-1)

  # feature generation
  b=np.ones(adj_matrix.shape[0])
  z=adj_matrix@b
  D=np.diag(z)
  L=D-adj_matrix
  feature_size = adj_matrix.shape[0]
  #node_features = torch.from_numpy(np.random.multivariate_normal(np.zeros(adj_matrix.shape[0]), np.linalg.pinv(L), feature_size).T.astype(np.float32))

  node_features = torch.from_numpy(adj_matrix).type(torch.float)

  if with_labels == False:
    num_classes = 1
    labels = torch.ones(adj_matrix.shape[0])
  else:
    num_classes = len(np.unique(G.labels))
    labels = torch.from_numpy(G.labels).type(torch.LongTensor)

  print(num_classes)

  data = Data(x = node_features, edge_index=edge_index, y = labels, num_nodes = adj_matrix.shape[0])

  G_nx = torch_geometric.utils.to_networkx(data, to_undirected=True)
  
  pos = {}
  for i, coord in enumerate(G.coords):
      pos[i] = coord
  #print(" pos ",len(pos))

  #G.plot(vertex_size=10)

  nx.draw(G_nx, pos=pos, node_size=10)
  #nx.draw_networkx_nodes(G_nx, pos=pos, node_size=10, node_color=G.labels)
  plt.show()

  return data, num_classes, feature_size, pos


def plot_coarsened_graphs(pos, P, adj_matrix, labels=False):
  new_pos = {}
  P = np.array(P)
  i = 0
  for row in P:
    non_zeros_indices = np.nonzero(row)
    values = [pos[key] for key in non_zeros_indices[0]]
    new_pos[i] = values[0]
    #new_pos[i] = np.sum(values,axis = 0)/len(values)
    i += 1
  print("total supernodes are ",i)

  edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
  #x = torch.tensor(node_features, dtype=torch.float).unsqueeze(-1)
  x = torch.from_numpy(adj_matrix).type(torch.float)
  data = Data(x=x, edge_index=edge_index, num_nodes = adj_matrix.shape[0])

  G_nx = torch_geometric.utils.to_networkx(data, to_undirected=True)

  #nx.draw(G_nx, pos=new_pos, node_size=10)
  if labels == False:
    nx.draw_networkx_nodes(G_nx, pos=new_pos, node_size=10)
  else:
    nx.draw_networkx_nodes(G_nx, pos=new_pos, node_size=10, node_color=labels)
  
  #plt.savefig("coarsened_ugc_scatter_50")
  plt.show()


def handling_scatter_alphabets_graphs(name):
  coords = sl.text_to_data(name, repeat=True, intensity = 5, rand=True, in_path=None)
  
  my_dict = {}

  for i in range(len(coords) - 1):
      new_list = []
      diff = max(coords[i][0]) - min(coords[i][0])
      if i == 0:
          min_value_x = min(coords[i][0])
          new_list.append([x - min_value_x  for x in coords[i][0]])
          max_value_x = max(coords[i][0]) - min_value_x
          min_value_x = 0
      else:
          min_value_x = max_value_x + 70
          max_value_x = min_value_x + diff
          new_list.append([x + (min_value_x - min(coords[i][0])) for x in coords[i][0]])
      
      new_list.append(coords[i][1])
      my_dict[i] = new_list

  new_list_x = []
  new_list_y = []

  for i in range(len(my_dict)):
      new_list_x.append(my_dict[i][0])
      new_list_y.append(my_dict[i][1])

  one_d_list_x = []
  one_d_list_y = []

  for sublist in new_list_x:
      for element in sublist:
          one_d_list_x.append(element)
          
  for sublist in new_list_y:
      for element in sublist:
          one_d_list_y.append(element)

  plt.scatter(one_d_list_x, one_d_list_y)
  plt.axis('off')
  # plt.savefig("A_original_scatter")
  # plt.title("Original")
  # plt.show()

  points = np.array([one_d_list_x,one_d_list_y]).T

  # create a NearestNeighbors object
  k = 10
  nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)

  # get the indices of the nearest neighbors for each point
  _, indices = nn.kneighbors(points)

  # create an empty graph
  G = nx.Graph()

  # add the nodes to the graph
  for i in range(len(points)):
      G.add_node(i, pos=points[i])

  # add the edges to the graph
  for i in range(len(points)):
      for j in indices[i][1:]:
          G.add_edge(i, j)

  num_nodes = G.number_of_nodes()
  
  W = np.zeros((num_nodes, num_nodes))

  for u, v in G.edges():
      # Assign a weight of 1 to each edge
      W[u, v] = 1
      W[v, u] = 1

  G.W = W
  G.coords = points

  # draw the graph
  pos = nx.get_node_attributes(G, 'pos')
  # nx.draw(G, pos = points, node_size = 10)
  # plt.show()

  return G


#################


if __name__ == "__main__":

  time1 = time.time()  
  args = parse_args()
  utils.fix_seeds(args.seed)
  device = torch.device("cpu")
  torch.cuda.empty_cache()

  if args.dataset_not_in_torch_geometric == True:
    '''Our dataset is not present on the torch_geometric datasets.
      Create Instance of the torch_geo from edge_index, label, node_feat.
    '''
    if args.edge_index_path == False or args.label_path == False or args.node_feat_path == False or args.num_classes == -1 or args.feature_size == -1:
      print("One or more required variable for creating Instance of Geometric dataset is missing. Please try again after giving information about following variable edge_index_path, label_path, node_feat_path, feature_size and num_classes")
      exit(1)
    
    new_dataset_hetro_node_feat = torch.load(args.node_feat_path)
    new_dataset_edge_index = torch.load(args.edge_index_path)
    new_dataset_hetro_label = torch.from_numpy(torch.load(args.label_path)).type(torch.LongTensor)

    data = Data(x=new_dataset_hetro_node_feat, edge_index = new_dataset_edge_index, y = new_dataset_hetro_label)
    num_classes = args.num_classes
    feature_size = args.feature_size
    print("done with new_dataset formation.")
  
  elif args.gsp_graphs == True:
    with_label = False
    
    if args.dataset == 'logo':
      G = pygsp.graphs.Logo()
    elif args.dataset == 'comet':
      G = pygsp.graphs.Comet()
    elif args.dataset == 'community':
       G = pygsp.graphs.Community()
    elif args.dataset == 'ring':
       G = pygsp.graphs.Ring()
    else:
      with_label = True
      G = pygsp.graphs.TwoMoons()

    data, num_classes, feature_size, pos = handling_gsp_graphs(G, with_label)
    print("done with fetching gsp_graphs")

  elif args.scatter_alphabets != "None":
    G = handling_scatter_alphabets_graphs(args.scatter_alphabets)
    print("done with handling_scatter_alphabets_graphs")
    data, num_classes, feature_size, pos = handling_gsp_graphs(G, False, True)
    print("done with fetching gsp_graphs")
    
    # exit(1)

  else:
    if args.dataset == 'karate':
      '''
      KarateClub nodes dont have features so we are generating its node features
      using its Laplacian's pseudo inverse see  karateClub_data_generation()
      for more details.
      '''
      dataset = KarateClub()
      karate_data_generation = 'deep_walk'

      if karate_data_generation == 'deep_walk':
        data, feature_size, num_classes = utils.karateClub_data_generation_deepwalk()
      else:
        data, feature_size, num_classes = utils.karateClub_data_generation()

    elif args.dataset == 'AMiner':
      # Heterogenous data
      dataset = AMiner(root = 'data/AMiner')
    
    elif args.dataset == 'OGB_MAG':
    # Heterogenous data
      dataset = OGB_MAG(root='./data', preprocess='metapath2vec')

    elif args.dataset == 'flickr':
      dataset = Flickr(root = 'data/Flickr')

    elif args.dataset == 'yelp':
      dataset = Yelp(root = 'data/Yelp')

    elif args.dataset == 'reddit':
      dataset = Reddit(root = 'data/Reddit')

    elif args.dataset == 'reddit2':
      dataset = Reddit2(root = 'data/Reddit')

    elif args.dataset == 'citeseer':
      dataset = Planetoid(root = 'data/CiteSeer', name = 'CiteSeer')

    elif args.dataset == 'cora':
      dataset = Planetoid(root = 'data/Cora', name = 'Cora')

    elif args.dataset == 'pubmed':
      dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')

    elif args.dataset == 'physics':
      dataset = Coauthor(root = 'data/Physics', name = 'Physics')
    
    elif args.dataset == 'dblp':
      dataset = CitationFull(root = 'newdata/DBLP', name = 'DBLP')

    elif args.dataset == 'cs':
      dataset = Coauthor(root = 'data/CS', name = 'CS')

    elif args.dataset == 'amazon':
      dataset = AmazonProducts(root = 'data/AmazonProducts')

    elif args.dataset == 'squirrel':
      num_classes = 5
      file_edge_index = 'heterophlic_data/edge_index_squirrel.pt'
      file_label = 'heterophlic_data/label_squirrel.pt'
      node_feat = 'heterophlic_data/node_feat_squirrel.pt'

      hetro_edge_index = torch.load(file_edge_index)
      hetro_label = torch.from_numpy(torch.load(file_label)).long()
      hetro_node_feat = torch.load(node_feat)
       
    elif args.dataset == 'chameleon':
      num_classes = 5
      file_edge_index = 'heterophlic_data/edge_index_chameleon.pt'
      file_label = 'heterophlic_data/label_chameleon.pt'
      node_feat = r"heterophlic_data\node_feat_cameleon.pt"
      

      hetro_edge_index = torch.load(file_edge_index)
      hetro_label = torch.from_numpy(torch.load(file_label)).long()
      hetro_node_feat = torch.load(node_feat)
       
    elif args.dataset == 'texas':
      num_classes = 5
      file_edge_index = 'heterophlic_data/edge_index_texas.pt'
      file_label = 'heterophlic_data/label_texas.pt'
      node_feat = 'heterophlic_data/node_feat_texas.pt'

      hetro_edge_index = torch.load(file_edge_index)
      hetro_label = torch.from_numpy(torch.load(file_label)).long()
      hetro_node_feat = torch.load(node_feat)

    elif args.dataset == 'cornell':
      num_classes = 5
      file_edge_index = 'heterophlic_data/edge_index_cornell.pt'
      file_label = 'heterophlic_data/label_cornell.pt'
      node_feat = 'heterophlic_data/node_feat_cornell.pt'

      hetro_edge_index = torch.load(file_edge_index)
      hetro_label = torch.from_numpy(torch.load(file_label)).long()
      hetro_node_feat = torch.load(node_feat)
       
    elif args.dataset == 'film':
      num_classes = 5
      file_data_path = r'heterophlic_data\film.mat'
      film_data = scipy.io.loadmat(file_data_path)
      hetro_edge_index = torch.from_numpy(film_data['edge_index']).long()
      hetro_label = torch.from_numpy(np.squeeze(film_data['label']))
      hetro_node_feat = torch.from_numpy(film_data['node_feat'])
    
    else:
      print("For now UGC don't support your mentioned dataset: ",args.dataset,". \nExiting......")
      exit(1)

    ### ----------------- getting the homophilic factor
    # same_class_edges = 0
    # total_edges = 0
    # for edge in dataset[0].edge_index.T:
    #     node1 = edge[0]
    #     node2 = edge[1]

        
    #     # Check if the nodes belong to the same class
    #     if dataset[0].y[node1] == dataset[0].y[node2]:
    #         same_class_edges += 1

    #     total_edges += 1

    # print(same_class_edges)
    # print(total_edges)
    # homophilic_fator = same_class_edges / total_edges
    # print("homophilic_fator ",homophilic_fator)
    ##-----------------------------------

    # distances = cdist(dataset[0].x, dataset[0].x, metric='euclidean')
    
    # # Sum of Euclidean distances
    # average_distance = np.mean(distances)

    # print('average_distance of Euclidean distances:', average_distance)


  if args.dataset not in ['karate', 'chameleon', 'squirrel', 'texas', 'cornell', 'film']:
    data = dataset[0]    
    num_classes = dataset.num_classes
    feature_size = dataset.num_features
  elif args.dataset != 'karate':
    # hetro_adj = to_dense_adj(hetro_edge_index)[0]
    data = Data(x=hetro_node_feat, edge_index = hetro_edge_index, y = hetro_label)
    feature_size = hetro_node_feat.shape[1]
  

  if args.full_dataset == True:
    time1 = time.time()
    incorrect_index = train_on_original_dataset(data,num_classes,feature_size,args.hidden_units,args.lr,args.decay,args.epochs)
    time2 = time.time()
    print("time taken to train GCN ", time2 - time1)
    # print(incorrect_index)
    # print(len(incorrect_index))
    exit(1)
      
    # if args.add_adj_to_node_features == True:
    #   g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
    #   #adding self loops
    #   # g_adj.fill_diagonal_(1)
      
    #   #Add random noise to increase the uniqueness of supernodes range of randomness should be small such that similarity of nodes still exist also it should not be too 
    #   #small else we will not be able to induce the uniqueness.
    #   epsilon = 0.01
    #   random_numbers = np.random.uniform(-epsilon, epsilon, g_adj.shape)
    #   g_adj = g_adj.numpy()
    #   # Replace non-zero entries in the array with random numbers
    #   g_adj[g_adj != 0] = random_numbers[g_adj != 0]
    #   g_adj = torch.matmul(torch.from_numpy(g_adj),torch.matmul(torch.from_numpy(g_adj),torch.matmul(torch.from_numpy(g_adj),torch.matmul(torch.from_numpy(g_adj),torch.from_numpy(g_adj)))))

    #   # alpha decides how much heterophily you want
    #   alpha = args.alpha
    
    # ###----------------------Can this be treated as a New Way for getting a heterophily measure
    # aplha_list = my_array = [alpha] * data.x.shape[0]
    # from numpy import linalg as LA

    # for i in range(len(incorrect_index)):
    #   for j in range(data.x.shape[0]):
    #     feature_i = data.x[incorrect_index[i]]
    #     feature_j = data.x[j]
    #     adj_i = g_adj[i]
    #     adj_j = g_adj[j]

    #     feature_dot = np.matmul(feature_j,feature_i)/(LA.norm(feature_i)+LA.norm(feature_j))
    #     adj_dot = np.matmul(adj_i,adj_j)/(np.nonzero(adj_i)[0].shape[0]+np.nonzero(adj_j)[0].shape[0])
    #     # print(np.nonzero(adj_i)[0].shape[0])

    #   # print("node ",i)
    #   # print("feature_dot ",np.abs(feature_dot)," adj_dot ",np.abs(adj_dot))
    #   # print(np.abs(np.abs(feature_dot) - np.abs(adj_dot)))
    #   aplha_list[i] = np.abs(np.abs(feature_dot) - np.abs(adj_dot))

    # print(aplha_list)
    # g_adj = torch.tensor(aplha_list)*g_adj


    # ###-----------------------


  if args.add_adj_to_node_features == True:
      data.x = (1-args.alpha)*data.x
      g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
      g_adj = args.alpha*g_adj
      
      data.x = torch.cat((data.x, g_adj), dim = 1)
      feature_size = feature_size + data.num_nodes
  else:
      print("UGC(only adjacency) no need of feature augumentation matrix")

  no_of_hash = args.number_of_projectors
  out_of_sample = args.out_of_sample
  hash_function = args.hash_function
  projectors_distribution = args.projectors_distribution

  ### split way 1
  # test_split_percent = 0.2
  # data = split(data,num_classes,test_split_percent) 
  ###


  ### split way 2
  num_nodes = data.num_nodes
  perm = torch.randperm(num_nodes)

  num_train = int(num_nodes * 0.6)
  num_val = int(num_nodes * 0.2)
  num_test = num_nodes - num_train - num_val

  data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
  data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
  data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

  data.train_mask[perm[:num_train]] = True
  data.val_mask[perm[num_train:num_train+num_val]] = True
  data.test_mask[perm[num_train+num_val:]] = True
  ###


  ### split way 3
  # train_rate = 0.6
  # val_rate = 0.2
  # percls_trn = int(round(train_rate*len(data.y)/num_classes))
  # val_lb = int(round(val_rate*len(data.y)))

  # data = utils.random_splits(data, num_classes, percls_trn, val_lb)
  ###

  time2 = time.time()
  
  # Bin_values = hashed_values(data, no_of_hash, feature_size,hash_function,out_of_sample,projectors_distribution)  
  Bin_values = hashed_values(data, no_of_hash, feature_size,hash_function,out_of_sample,projectors_distribution,to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0])  
  
  time3 = time.time()
  
  
  list_bin_width = allocate_list_bin_width(args.dataset,[args.ratio],args.hash_function,args.scatter_alphabets)
  
  summary_dict = {}
  summary_dict = partition(list_bin_width,Bin_values,no_of_hash)
  temp_time4 = time.time()
  print("time taken in coarsening(find partition matrix)",temp_time4-time2)

  he_error_list = []
  ree_error_list = []
  dirichlet_energy_list = []

  for bin_width in list_bin_width:
      time4 = time.time()
      current_bin_width_summary = summary_dict[bin_width]
      values = current_bin_width_summary.values()
      unique_values = set(values)
      rr = 1 - len(unique_values)/len(values)
      print(f'Graph reduced by: {rr*100} percent.\nWe now have {len(unique_values)} supernode, starting nodes were: {len(values)}')
      dict_blabla ={}
      C_diag = torch.zeros(len(unique_values))#, device= device)
      help_count = 0
      
      for v in unique_values:
          C_diag[help_count],dict_blabla[help_count] = utils.get_key(v, current_bin_width_summary)
          help_count += 1

      P_hat = torch.zeros((data.num_nodes, len(unique_values)))#, device= device)
      zero_list = torch.ones(len(unique_values), dtype=torch.bool)
      
      for x in dict_blabla:
          if len(dict_blabla[x]) == 0:
            print("zero element in this supernode",x)
          for y in dict_blabla[x]:
              P_hat[y,x] = 1
              zero_list[x] = zero_list[x] and (not (data.train_mask)[y])
            
      P_hat = P_hat.to_sparse()
      #dividing by number of elements in each supernode to get average value 
      P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))
      
      features =  data.x.to(device = device).to_sparse()


      #-----------------------different_bin_frequency
      # print("Check")
      # pairwise_distances = []
      # print(P_hat.shape)
      # counter = 0
      # empty_supernode = 0
      # projection_sum = 0
      # for row in P_hat.to_dense().T:
      #   counter += 1
      #   if counter%500 == 0:
      #    print(counter)
      #   indices_of_ones = np.where(row > 0)[0]
      #   if indices_of_ones != []:
      #     current_supernode_feat = features.to_dense()[indices_of_ones]
      #     #projection_sum += utils.projection_distance(current_supernode_feat)

      #     #values = [sum(row) for row in utils.projection_distance(current_supernode_feat)]
      #     pairwise_distances.append(list(pdist(utils.projection_distance(current_supernode_feat), metric='euclidean')))
      #   else:
      #     empty_supernode += 1

      # print("empty_supernodes !!!!!!!!!!!!!! alert need to check it",empty_supernode)
      # pairwise_distances = list(chain.from_iterable(pairwise_distances))
      # print(len(pairwise_distances))
      # print("projection_sum ",projection_sum)
          
      # utils.different_bin_frequency(pairwise_distances)


###----------------LSH distance anylsis-------------- 

      # print("Check")
      # distance_matrix_file = "distance_matrix_dblp.csv"
      # if os.path.exists(distance_matrix_file):
      #     # If the file exists, load the distance matrix from the file
      #     distance_matrix = np.loadtxt(distance_matrix_file, delimiter=",")
      # else:
      #     distance_matrix = utils.pair_wise_distance(data.x)
      #     np.savetxt(distance_matrix_file, distance_matrix, delimiter=",")
      
      # threshold_distance = 1
      # same_bin_counter = 0
      # different_bin_counter = 0

      # # Iterate through pairs of original nodes
      # print(P.to_dense().shape)
      # for i in range(P_hat.to_dense().shape[0]):
      #     for j in range(i+1, P_hat.to_dense().shape[0]):  # Avoid comparing nodes with themselves
      #         distance = distance_matrix[i, j]
              
      #         # Check if the distance is below the threshold
      #         if distance < threshold_distance:
      #             # Check if the nodes belong to the same supernode using the partition matrix
      #             # print(np.where(P.to_dense()[i, :] > 0.0),np.where(P.to_dense()[j, :] > 0.0))
      #             if np.where(P.to_dense()[i, :] > 0.0) == np.where(P.to_dense()[j, :] > 0.0):
      #                 same_bin_counter += 1
      #             else:
      #                 different_bin_counter += 1
      
      # print("same_bin_counter ",same_bin_counter)
      # print("different_bin_counter ",different_bin_counter)

      #----------------------------------



      # cor_feat : features of supernodes by averaging out all the features values of child nodes
      cor_feat = (torch.sparse.mm((torch.t(P)), features.to_dense()))#.to_sparse()
      i = data.edge_index
      v = torch.ones(data.edge_index.shape[1])
      shape = torch.Size([data.x.shape[0],data.x.shape[0]])
      g_adj_tens = torch.sparse.FloatTensor(i, v, torch.Size(shape))#.to(device = device)
      g_coarse_adj = torch.sparse.mm(torch.t(P_hat) , torch.sparse.mm( g_adj_tens , P_hat))
      
      C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))
      #print("number of edges in the coarsened graph ",np.count_nonzero(g_coarse_adj.to_dense().to('cpu').numpy())/2)

      g_coarse_dense = g_coarse_adj.to_dense().to('cpu').numpy() + C_diag_matrix - np.identity(C_diag_matrix.shape[0], dtype = np.float32)
      
      
      if args.induce_adverserial_edges == True:
        print("You have decided to induce adverserial edges into your graph\n")
        for i in range((int)(np.shape(g_coarse_dense)[0]*0.1)):
          for j in range((int)(np.shape(g_coarse_dense)[0]*0.1)):
            g_coarse_dense[i][j] = 0
     
      edge_weight = g_coarse_dense[np.nonzero(g_coarse_dense)]
      edges_src = torch.from_numpy((np.nonzero(g_coarse_dense))[0])
      edges_dst = torch.from_numpy((np.nonzero(g_coarse_dense))[1])
      edge_index_corsen = torch.stack((edges_src, edges_dst))
      edge_features = torch.from_numpy(edge_weight)

      #------------------
      ## Epsilion bounds
      
      # epsilion_bound = utils.get_smooth_features(data.edge_index, P_hat, data.x.numpy())
      # print("epsilion_bound ", epsilion_bound)
      # exit(1)
      #------------------

      # -----------
      if args.gsp_graphs == True or args.scatter_alphabets != "None":
        plot_coarsened_graphs(pos, P.T, g_coarse_dense)#, labels=labels_coarse)
        print("plot_coarsened_graphs ")
        exit(1)
      #-------------


      if args.calculate_spectral_errors == True:
        if data.x.size(0) < 100:
          number_of_eigen_vectors = (int)(data.x.size(0)/2)
        else:
          number_of_eigen_vectors = 100
        
        spectral_properties.eigen_error(data.edge_index, edge_index_corsen, edge_features, number_of_eigen_vectors)

      Y = np.array(data.y.cpu())
      Y = utils.one_hot(Y,num_classes)#.to(device)
      Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)])#.to(device)
      labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1)#.to(device)

      # deleting unused variables
      del C_diag_matrix
      del g_coarse_adj
      del edge_weight
      del edges_dst
      del i
      del v

      data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
      data_coarsen.edge_attr = edge_features

      # Persist the coarsened graph (generic version of the per-dataset block
      # below, which upstream leaves commented out). Also saves the partition
      # matrix P_hat, i.e. the node -> supernode assignment.
      if args.save_coarsened:
          import os as _os
          _os.makedirs('coarsened_data_UGC', exist_ok=True)
          _tag = f'{args.dataset}_ratio{args.ratio}_alpha{args.alpha}'
          torch.save(data_coarsen, f'coarsened_data_UGC/{_tag}_coarsened.pth')
          torch.save(P_hat, f'coarsened_data_UGC/{_tag}_Phat.pth')
          print(f'[saved] coarsened_data_UGC/{_tag}_coarsened.pth '
                f'(x={tuple(cor_feat.shape)}, edges={edge_index_corsen.shape[1]})')
          print(f'[saved] coarsened_data_UGC/{_tag}_Phat.pth '
                f'(partition matrix {tuple(P_hat.shape)})')

      #### neurIPS rebuttal
      # if args.dataset == 'cora':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/cora.pth')
      # elif args.dataset == 'citeseer':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/citeseer.pth')
      # elif args.dataset == 'pubmed':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/pubmed.pth')
      # elif args.dataset == 'physics':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/physics.pth')
      # elif args.dataset == 'dblp':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/dblp.pth')
      # elif args.dataset == 'squirrel':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/squirrel.pth')
      # elif args.dataset == 'chameleon':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/cchameleonora.pth')
      # elif args.dataset == 'texas':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/texas.pth')
      # elif args.dataset == 'film':
      #   torch.save(data_coarsen, 'coarsened_data_UGC/film.pth')

      # # exit(1)
      # ##

      
      if args.tsne_visualization == True:
        original_tsne_graph_name = 'results_and_plots/tsne_original_' + args.dataset 
        
        utils.t_sne_visualize_graph(data.x,data.y,original_tsne_graph_name)
        coarsen_tsne_graph_name = 'results_and_plots/tsne_coarsen_' + args.dataset
        utils.t_sne_visualize_graph(data_coarsen.x.to_dense(),data_coarsen.y,coarsen_tsne_graph_name)

      # data.edge_index, edge_index_corsen, edge_features
      if args.calculate_spectral_errors == True:
        he_error = spectral_properties.hyperbolic_error(np.array(P_hat.to_dense()).T,data.edge_index,edge_index_corsen,edge_features,np.array(data.x))
        he_error_list.append(he_error)
        print("check hyperbolic error",he_error)
        
        # eigen_plot_name = 'results_and_plots/' + args.dataset + '_' + (str)(math.floor(rr*100))
        # # spectral_properties.plot_most_significant_eigen_values(100,data.edge_index,edge_index_corsen,edge_features,eigen_plot_name)
        
        # re_construct_error = spectral_properties.reconstruction_error(data.num_nodes,np.array(P_hat.to_dense()).T,data.edge_index,edge_index_corsen,edge_features)
        # ree_error_list.append(re_construct_error)
        # print("re_construction error ",re_construct_error)
        
        # diri_energy = spectral_properties.dirichlet_energy(np.array(P_hat.to_dense()),data.edge_index,edge_index_corsen,edge_features,np.array(data.x),np.array(cor_feat.to_dense()))
        # dirichlet_energy_list.append(diri_energy)
        # print("dirichlet_energy error ",diri_energy)
      

      #this is main g_coarse_adj use it to visualize supernodes
      if args.visualize_graph == True:
        original_graph_name = 'results_and_plots/original_' + args.dataset
        pos, _, avg_degree = utils.visualize_graph(data.edge_index,data.num_nodes,data.y,original_graph_name)
        
        new_pos = {}
        i = 0
        for row in P.T:
          non_zeros_indices = np.array(np.nonzero(row))
          values = [pos[key] for key in non_zeros_indices[0]]
          new_pos[i] = values[0]
          #new_pos[i] = np.sum(values,axis = 0)/len(values)
          i += 1
        print("total supernodes are ",i)
        
        coarsen_graph_name = 'results_and_plots/coarsen_' + args.dataset + '_' + (str)(math.floor(rr*100))
        _, g, _ = utils.visualize_graph(edge_index_corsen,len(unique_values),data_coarsen.y,coarsen_graph_name,new_pos)
 
      time5 = time.time()
      print('diff b/w t5 and t4 {}'.format(time5-time4))

      all_acc = []
      num_run = 1

      time_taken_to_train_gcn = []
      for i in range(num_run):
        global_best_val = 0
        global_best_test = 0
        best_val_acc = 0
        best_epoch = 0

        hidden_units = args.hidden_units
        learning_rate = args.lr
        decay = args.decay
        epochs = args.epochs
        
        if args.model_type == 'gin':
           model = GIN.GIN(feature_size, hidden_units, num_classes)
        elif args.model_type == 'sage':
          model = GraphSage.GraphSAGE(feature_size, hidden_units, num_classes)
        elif args.model_type == 'gat':
          model = GAT.GAT(feature_size, hidden_units, num_classes)
        elif args.model_type == 'ugc':
          model = APPNP.Net(feature_size, hidden_units, num_classes)
        elif args.model_type == '3wl':
          model = WL_base_model.WL_BaseModel(feature_size, hidden_units, num_classes)
        else:
          model = GCN.GCN_(feature_size, hidden_units, num_classes)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        data = data.to(device)
        data_coarsen = data_coarsen.to(device)
        edge_weight = torch.ones(data_coarsen.edge_index.size(1))
        decay = decay
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)

        for epoch in range(epochs):
            optimizer.zero_grad()

            if args.model_type not in ['gcn','3wl']:
              out = model(data_coarsen.x, data_coarsen.edge_index)
            elif args.model_type == '3wl':
               out = model(data_coarsen.x)
            else:
              out = model(data_coarsen.x, data_coarsen.edge_index,data_coarsen.edge_attr.float())

            # out = model(data_coarsen.x, data_coarsen.edge_index,data_coarsen.edge_attr.float())
            # out = model(data_coarsen.x, data_coarsen.edge_index) 
            
            pred = out.argmax(1)
            criterion = torch.nn.NLLLoss()
            # print(out.shape)
            loss = criterion(out[~zero_list], data_coarsen.y[~zero_list]) 
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            # print(data.x.dtype)
            # print(data.edge_index.dtype)
            # print(data.edge_attr.dtype)
            val_acc = val(model,data)

            if best_val_acc < val_acc:
                torch.save(model, 'best_model.pt')
                best_val_acc = val_acc
                best_epoch = epoch
          
            if epoch % 100 == 0:
                print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))

        time6 = time.time()
        print('diff b/w t6 and t5 {}'.format(time6-time5))
        time_taken_to_train_gcn.append(time6-time5)
        model = torch.load('best_model.pt')
        model.eval()
        data = data.to(device)
        
        if args.model_type not in ['gcn','3wl']:
          pred = model(data.x, data.edge_index).argmax(dim=1)
        elif args.model_type == '3wl':
          pred = model(data.x).argmax(dim=1)
        else:
          pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
      
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        
        acc = int(correct) / int(data.test_mask.sum())

        time7 = time.time()
        #print('diff b/w t7 and t5 {}'.format(time7-time5))
        all_acc.append(acc)
        
        # if t_sne and other visualizations take time it is better to use this limited visualization to get 
        # the gist of data

        # np.random.seed(432)
        # temp = random.sample(range(0, data.num_nodes), 2000)
        #print(temp)
        #t_sne_visualize_graph(data.x[temp],data.y[temp],"tsne_physics_limited")
        #t_sne_visualize_graph(data.x[temp],pred[temp],"tsne_physics_coarsened_50_limited")
        #t_sne_visualize_graph(data_coarsen.x.to_dense(),data_coarsen.y,"tsne_physics_only_ugc_30")
      
      print("ratio ",rr)
      print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))
      print('ave_time: {:.4f}'.format(np.mean(time_taken_to_train_gcn)), '+/- {:.4f}'.format(np.std(time_taken_to_train_gcn)))
      print("he_error_list ",he_error_list)
      print("ree_error_list ",ree_error_list)
      print("dirichlet_energy_list ",dirichlet_energy_list)