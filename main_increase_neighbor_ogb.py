import sys
import os
import datetime

import numpy as np
import torch
import pickle as pkl

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask

import matplotlib.pyplot as plt

import gnn
import train_test_neighbor as train_test
from aux_functions import return_node_idx

# TO DO: 
# 5) In the other script (neighbor sampling): do the same that I do here
""
""

# Check devices
#if torch.cuda.is_available():
#    device = 'cuda:0'
#else:
device = 'cpu'

#limit_epoch = 0

# total arguments
n0 = 500#int(sys.argv[1])
n_epochs_per_n = 10#int(sys.argv[2])

figSize = 5
plt.rcParams.update({'font.size': 16})

thisFilename = 'ogb_neigh_' + str(n0) + '_' + str(n_epochs_per_n) # This is the general name of all related files

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
        {'batch_size': 32, 'epochs': 50, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.0001},
    ]:
        args = objectview(args)

dataset_vector = []
loader_vector = []
val_loader_vector = []
another_loader_vector = []
another_val_loader_vector = []

#n0 = 800
n_epochs = args.epochs
#n_increases = n_epochs
#n_epochs_per_n = 1#int(n_epochs/n_increases)
n_increases = int(n_epochs/n_epochs_per_n)
#increase_rate = 20

for args2 in [
        {'batch_size': 32, 'epochs': n_epochs_per_n, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.0001},
    ]:
        args2 = objectview(args2)

# Loss

loss = torch.nn.NLLLoss()

# Data

dataset = PygNodePropPredDataset(name='ogbn-mag')
rel_data = dataset[0]

split_idx = dataset.get_idx_split()

train_idx = split_idx['train']['paper'].to(device)
val_idx = split_idx['valid']['paper'].to(device)
test_idx = split_idx['test']['paper'].to(device)

nTrain = torch.sum(train_idx).item()
nVal = torch.sum(val_idx).item()
nTest = torch.sum(test_idx).item()

m = rel_data.x_dict['paper'].shape[0]

# We are only interested in paper <-> paper relations.
data = Data(
    x=rel_data.x_dict['paper'],
    edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
    y=rel_data.y_dict['paper'],
    train_mask=index_to_mask(train_idx,size=m),
    val_mask=index_to_mask(val_idx,size=m),
    test_mask=index_to_mask(test_idx,size=m))

data = T.ToUndirected()(data)

edge_list = data.edge_index
F0 = rel_data.x_dict['paper'].shape[1]
C = dataset.num_classes
data = data.to(device)

# GNN models

modelList = dict()

F = [F0, 64, 32]
MLP = [32, C]
K = [2, 2]

GNN = gnn.GNN('gnn', F, MLP, True, K)
#modelList['GNN'] = GNN

SAGE = gnn.GNN('sage', F, MLP, True)
SAGE = SAGE.to(device)
modelList['SAGE'] = SAGE

GCN = gnn.GNN('gcn', F, MLP, True)
GCN = GCN.to(device)
modelList['GCN'] = GCN

SAGELarge = gnn.GNN('sage', F, MLP, True)
SAGELarge = SAGELarge.to(device)
modelList['SAGE full'] = SAGELarge

GCNLarge = gnn.GNN('gcn', F, MLP, True)
GCNLarge = GCNLarge.to(device)
modelList['GCN full'] = GCNLarge

GNNLarge = gnn.GNN('gnn', F, MLP, True, K)
#modelList['GNN full'] = GNNLarge

color = {}
color['SAGE'] = 'orange'
color['GCN'] = 'mediumpurple'
color['GNN'] = 'dodgerblue'

# Trasferability    
dataset_transf = [data]
another_test_loader = NeighborLoader(dataset_transf[0], num_neighbors=[-1]*(len(F)-1), 
                                     batch_size=nTest, input_nodes = data['test_mask'], shuffle=False)
m = n0
print(n_increases)
for i in range(n_increases+1):
    #epoch = i*n_epochs_per_n
    #if epoch <= limit_epoch:
    #    m = n0 + increase_rate*i
    print(i)
    idx = return_node_idx(edge_list,m)
    sampledData = data.subgraph(torch.tensor(idx).to(device))#data.subgraph(torch.randint(0, data.num_nodes, (m,)))
    # fix here; val has to be on large graph
    dataset = [sampledData]
    dataset_vector.append(dataset)
    
    loader = NeighborLoader(sampledData, num_neighbors=[32]*(len(F)-1), 
                            batch_size=args.batch_size, input_nodes = sampledData['train_mask'], shuffle=False)
    val_loader = NeighborLoader(sampledData, num_neighbors=[32]*(len(F)-1), 
                                batch_size=nVal, input_nodes = sampledData['val_mask'], shuffle=False)
    another_loader = NeighborLoader(dataset_transf[0], num_neighbors=[32]*(len(F)-1), 
                                batch_size=args.batch_size, input_nodes = dataset_transf[0]['train_mask'], shuffle=False)
    another_val_loader = NeighborLoader(dataset_transf[0], num_neighbors=[32]*(len(F)-1), 
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

#plt.show()

print()

pkl.dump(time_dict, open(os.path.join(saveDir,'time_per_epoch.p'), "wb"))
pkl.dump(test_acc_dict, open(os.path.join(saveDir,'test_accs_full.p'), "wb"))
pkl.dump(best_accs, open(os.path.join(saveDir,'best_accs.p'), "wb"))
