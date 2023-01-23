import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy
import pickle as pkl

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt

import gnn
import train_test

# TO DO: 1) Look at Juan's plot; idea is to shown gains in speed but also 
# better performace wrt transferability/comparable performance wrt large graph training
# 2) Change validation graph
# 3) Add "small" (transferability) model and large (full-size) model
# 4) Add final_model
# 5) In the other script (neighbor sampling): do the same that I do here
# 6) Keep track of training time
""
""

# Aux class

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

for args in [
        {'batch_size': 32, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
    ]:
        args = objectview(args)

dataset_vector = []
loader_vector = []
another_loader_vector = []

n_epochs = args.epochs
n_increases = 20
n_epochs_per_n = int(n_epochs/n_increases)
increase_rate = 50
n0 = 2000

for args2 in [
        {'batch_size': 32, 'epochs': n_epochs_per_n, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
    ]:
        args2 = objectview(args2)

# Loss

loss = torch.nn.NLLLoss()

# Data

dataset = Planetoid(root='/tmp/cora', name='Cora', split='public')
F0 = dataset.num_node_features
C = dataset.num_classes
data = dataset.data 
m = n0 + increase_rate*(n_increases)
data = data.subgraph(torch.randint(0, data.num_nodes, (m,)))

# Trasferability    
dataset_transf = [data]
another_test_loader = DataLoader(dataset_transf, batch_size=args.batch_size, shuffle=False)

for i in range(n_increases+1):
    m = n0 + increase_rate*i
    sampledData = data.subgraph(torch.randint(0, data.num_nodes, (m,)))
    # fix here; val has to be on large graph
    dataset = [sampledData]
    dataset_vector.append(dataset)
    loader_vector.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    another_loader_vector.append(another_test_loader)


# GNN models

modelList = dict()

F = [F0, 64, 32]
MLP = [32, C]
K = [5, 5]

#GNN = gnn.GNN('gnn', F, MLP, True, K)
#modelList.append(GNN)

SAGE = gnn.GNN('sage', F, MLP, True)
modelList['sage'] = SAGE

GCN = gnn.GNN('gcn', F, MLP, True)
modelList['gcn'] = GCN

SAGELarge = gnn.GNN('sage', F, MLP, True)
modelList['sage_large'] = SAGELarge

GCNLarge = gnn.GNN('gcn', F, MLP, True)
modelList['gcn_large'] = GCNLarge

loader_vector_dict = dict()
loader_vector_dict['sage'] = loader_vector
loader_vector_dict['gcn'] = loader_vector
loader_vector_dict['sage_large'] = another_loader_vector
loader_vector_dict['gcn_large'] = another_loader_vector

test_acc_dict = dict()
time_dict = dict()
best_accs = dict()

# Training and testing

for model_key, model in modelList.items():
    
    print('Training ' + model_key + '...')
    
    best_model = model
    best_acc = 0
    for count, loader in enumerate(loader_vector_dict[model_key]):
        #loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        best_acc_old = best_acc
        best_model_old = best_model
        test_accs, losses, best_model, last_model, best_acc, test_loader, training_time = train_test.train(loader, another_test_loader, model, loss, args2) 
        if count == 0:
            test_accs_full = test_accs
            total_time = training_time
        else:
            test_accs_full += test_accs
            total_time += training_time
        if best_acc < best_acc_old:
            best_model = best_model_old
            best_acc = best_acc_old
        
        print("Maximum validation set accuracy: {0}".format(max(test_accs)))
        print("Minimum loss: {0}".format(min(losses)))
        
    time_dict[model_key] = total_time
    test_acc_dict[model_key] = test_accs_full
    best_accs[model_key] = best_acc
    # Run test for our best model to save the predictions!
    print(train_test.test(test_loader, best_model, is_validation=False, save_model_preds=True))
    # Run test for our last model to save the predictions!
    print(train_test.test(test_loader, last_model, is_validation=False, save_model_preds=True))

    # Run test for our best model to save the predictions!
    print(train_test.test(another_test_loader, best_model, is_validation=False, save_model_preds=True))
    # Run test for our last model to save the predictions!
    print(train_test.test(another_test_loader, last_model, is_validation=False, save_model_preds=True))

    print()

    plt.title('Cora')
    #plt.plot(losses, label="training loss" + " - " + model_key)
    if 'large' in model_key:
        plt.plot(best_acc*np.ones(len(test_accs_full)), label="test accuracy" + " - " + model_key)
    else:
        plt.plot(test_accs_full, label="test accuracy" + " - " + model_key)

plt.legend()
plt.show()

