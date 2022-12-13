import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt

import gnn
import train_test_neighbor as train_test

""
""

# Aux class

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

# Data

dataset = Planetoid(root='/tmp/pubmed', name='PubMed', split='random')
F0 = dataset.num_node_features
C = dataset.num_classes
data = dataset.data # Save it to have the same test samples in the transferability test
save_test_mask = data.test_mask
m = 2000
sampledData = data.subgraph(torch.randint(0, data.num_nodes, (m,)))
dataset = [sampledData]

dataset_transf = Planetoid(root='/tmp/pubmed', name='PubMed', split='random')
data = dataset_transf.data
data.test_mask = save_test_mask
dataset_transf = [data]

# GNN models

modelList = []

F = [F0, 64, 32]
MLP = [32, C]
K = [5, 5]

#GNN = gnn.GNN('gnn', F, MLP, True, K)
#modelList.append(GNN)

SAGE = gnn.GNN('sage', F, MLP, True)
modelList.append(SAGE)

GCN = gnn.GNN('gcn', F, MLP, True)
modelList.append(GCN)

# Loss

loss = torch.nn.NLLLoss()
for args in [
        {'batch_size': 128, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001},
    ]:
        args = objectview(args)

# Training and testing

nVal = torch.sum(dataset[0]['val_mask']).item()
nTest = torch.sum(dataset_transf[0]['test_mask']).item()


for model in modelList:
    
    loader = NeighborLoader(dataset[0], num_neighbors=[32]*(len(F)-1), batch_size=args.batch_size, input_nodes = dataset[0]['train_mask'], shuffle=False)
    
    val_loader = NeighborLoader(dataset[0], num_neighbors=[32]*(len(F)-1), batch_size=nVal, input_nodes = dataset[0]['val_mask'], shuffle=False)

    test_accs, losses, best_model, best_acc, test_loader = train_test.train(loader, val_loader, model, loss, args) 

    print("Maximum validation set accuracy: {0}".format(max(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))

    # Run test for our best model to save the predictions!
    print(train_test.test(test_loader, best_model, is_validation=False, save_model_preds=True))

    # Trasferability
    another_test_loader = NeighborLoader(dataset_transf[0], num_neighbors=[30]*(len(F)-1), batch_size=nTest, input_nodes = dataset_transf[0]['test_mask'], shuffle=False)

    # Run test for our best model to save the predictions!
    print(train_test.test(another_test_loader, best_model, is_validation=False, save_model_preds=True))

    print()

    plt.title('Pubmed')
    plt.plot(losses, label="training loss" + " - " + model.type)
    plt.plot(test_accs, label="test accuracy" + " - " + model.type)

plt.legend()
plt.show()
