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
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import matplotlib.pyplot as plt

import gnn
import train_test

""
""

modelList = []

# GNN models

dataset = Planetoid(root='/tmp/cora', name='Cora')

F = [dataset.num_node_features, 32]
MLP = [32, dataset.num_classes]

SAGE = gnn.GNN('sage', F, MLP)
modelList.append(SAGE)

GCN = gnn.GNN('gcn', F, MLP)
modelList.append(GCN)

# Loss

loss = torch.nn.NLLLoss

# Training and testing

for model in modelList:
    
    dataset = Planetoid(root='/tmp/cora', name='Cora')
    
    test_accs, losses, best_model, best_acc, test_loader = train_test.train(dataset, model, loss) 

    print("Maximum test set accuracy: {0}".format(max(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))

    # Run test for our best model to save the predictions!
    train_test.test(test_loader, best_model, is_validation=False, save_model_preds=True, model_type=model)
    print()

    plt.title(dataset.name)
    plt.plot(losses, label="training loss" + " - " + args.model_type)
    plt.plot(test_accs, label="test accuracy" + " - " + args.model_type)
plt.legend()
plt.show()
