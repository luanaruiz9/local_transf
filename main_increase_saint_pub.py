# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:13:38 2023

@author: Luana Ruiz
"""

import os
import datetime

import numpy as np
import torch
import pickle as pkl

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader, GraphSAINTNodeSampler, GraphSAINTEdgeSampler,  GraphSAINTRandomWalkSampler

import matplotlib.pyplot as plt

import gnn
import train_test_saint as train_test

# TO DO: 
# 5) In the other script (neighbor sampling): do the same that I do here
""
""

figSize = 5
plt.rcParams.update({'font.size': 16})

limit_epoch = 0

thisFilename = 'pubmed_saint_edge_3000' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) 

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + today
# Create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# Aux class

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        
for args in [
        {'batch_size': 256, 'epochs': 1000, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001}, # node is 1e-3
    ]:
        args = objectview(args)

dataset_vector = []
loader_vector = []
val_loader_vector = []
another_loader_vector = []
another_val_loader_vector = []

n_epochs = args.epochs
n_increases = n_epochs
n_epochs_per_n = int(n_epochs/n_increases)
increase_rate = 100
n0 = 3000

for args2 in [
        {'batch_size': 256, 'epochs': n_epochs_per_n, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001}, # node is 1e-3
    ]:
        args2 = objectview(args2)

# Loss

loss = torch.nn.NLLLoss()

# Data

dataset = Planetoid(root='/tmp/pubmed', name='PubMed', split='full')
F0 = dataset.num_node_features
C = dataset.num_classes
data = dataset.data 
m = data.num_nodes
data = data.subgraph(torch.randint(0, data.num_nodes, (m,)))
nVal = torch.sum(dataset[0]['val_mask']).item()

# GNN models

modelList = dict()

F = [F0, 64, 32]
MLP = [32, C]
K = [2, 2]

GNN = gnn.GNN('gnn', F, MLP, True, K)
#modelList['GNN'] = GNN

SAGE = gnn.GNN('sage', F, MLP, True)
modelList['SAGE'] = SAGE

GCN = gnn.GNN('gcn', F, MLP, True)
modelList['GCN'] = GCN

SAGELarge = gnn.GNN('sage', F, MLP, True)
modelList['SAGE full'] = SAGELarge

GCNLarge = gnn.GNN('gcn', F, MLP, True)
modelList['GCN full'] = GCNLarge

GNNLarge = gnn.GNN('gnn', F, MLP, True, K)
#modelList['GNN full'] = GNNLarge

color = {}
color['SAGE'] = 'firebrick'
color['GCN'] = 'steelblue'
color['GNN'] = 'dodgerblue'

n_neigh = -1

# Trasferability    
dataset_transf = [data]
nTest = torch.sum(dataset_transf[0]['test_mask']).item()
another_test_loader = NeighborLoader(dataset_transf[0], num_neighbors=[-1]*(len(F)-1), 
                                     batch_size=nTest, input_nodes = dataset_transf[0]['test_mask'], shuffle=False)

for i in range(n_increases+1):
    epoch = i*n_epochs_per_n
    if epoch <= limit_epoch:
        m = n0 + increase_rate*i
    sampledData = data.subgraph(torch.randint(0, data.num_nodes, (m,)))
    # fix here; val has to be on large graph
    dataset = [sampledData]
    dataset_vector.append(dataset)
    
    loader = GraphSAINTEdgeSampler(sampledData, batch_size=args.batch_size, shuffle=False)
    val_loader = NeighborLoader(sampledData, num_neighbors=[n_neigh]*(len(F)-1), 
                                batch_size=nVal, input_nodes = sampledData['val_mask'], shuffle=False)
    another_loader = GraphSAINTEdgeSampler(dataset_transf[0], batch_size=args.batch_size, shuffle=False)
    another_val_loader = NeighborLoader(dataset_transf[0], num_neighbors=[n_neigh]*(len(F)-1), 
                                batch_size=nVal, input_nodes = dataset_transf[0]['val_mask'], shuffle=False)
    
    loader_vector.append(loader)
    val_loader_vector.append(val_loader)
    another_loader_vector.append(another_loader)
    another_val_loader_vector.append(another_val_loader)

loader_vector_dict = dict()
val_loader_vector_dict = dict()
loader_vector_dict['SAGE'] = loader_vector
loader_vector_dict['GCN'] = loader_vector
loader_vector_dict['GNN'] = loader_vector
val_loader_vector_dict['SAGE'] = val_loader_vector
val_loader_vector_dict['GCN'] = val_loader_vector
val_loader_vector_dict['GNN'] = val_loader_vector
loader_vector_dict['SAGE full'] = another_loader_vector
loader_vector_dict['GCN full'] = another_loader_vector
loader_vector_dict['GNN full'] = another_loader_vector
val_loader_vector_dict['SAGE full'] = another_val_loader_vector
val_loader_vector_dict['GCN full'] = another_val_loader_vector
val_loader_vector_dict['GNN full'] = another_val_loader_vector

test_acc_dict = dict()
time_dict = dict()
best_accs = dict()

fig1, fig_last = plt.subplots(figsize=(1.4*figSize, 1*figSize))
fig2, fig_best = plt.subplots(figsize=(1.4*figSize, 1*figSize))

# Training and testing

for model_key, model in modelList.items():
    
    print('Training ' + model_key + '...')
    
    best_model = model
    best_acc = 0
    for count, loader in enumerate(loader_vector_dict[model_key]):
        #loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        best_acc_old = best_acc
        best_model_old = best_model
        test_accs, losses, best_model, last_model, best_acc, test_loader, training_time = train_test.train(loader, val_loader, model, loss, args2) 
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
        
    time_dict[model_key] = total_time/n_epochs
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

    #plt.plot(losses, label="training loss" + " - " + model_key)
    if 'SAGE' in model_key:
        col = color['SAGE']
    elif 'GCN' in model_key:
        col = color['GCN']
    else:
        col = color['GNN']
        
    if 'full' in model_key:
        fig_last.plot(test_accs_full[-1]*np.ones(len(test_accs_full)), '--', color=col, label=model_key)
        fig_best.plot(best_acc*np.ones(len(test_accs_full)), '--', color=col, label=model_key)
    else:
        fig_last.plot(test_accs_full, color=col, alpha=0.5, label=model_key)
        fig_best.plot(test_accs_full, color=col, alpha=0.5, label=model_key)

#fig_last.axvline(x = limit_epoch, alpha=0.8, linestyle=':', color = 'black')        
fig_last.set_ylabel('Accuracy')
fig_last.set_xlabel('Epochs')
fig_last.legend()
fig1.savefig(os.path.join(saveDir,'accuracies_last.pdf'))

#fig_best.axvline(x = limit_epoch, alpha=0.8, linestyle=':', color = 'black')
fig_best.set_ylabel('Accuracy')
fig_best.set_xlabel('Epochs')
fig_best.legend()
fig2.savefig(os.path.join(saveDir,'accuracies_best.pdf'))

plt.show()

print()

pkl.dump(time_dict, open(os.path.join(saveDir,'time_per_epoch.p'), "wb"))
pkl.dump(test_acc_dict, open(os.path.join(saveDir,'test_accs_full.p'), "wb"))
pkl.dump(best_accs, open(os.path.join(saveDir,'best_accs.p'), "wb"))
