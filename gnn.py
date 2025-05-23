import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv

def LSIGF(weights, S, x):
    '''
    weights is a list of length k, with each element of shape d_in x d_out
    S is N x N, sparse matrix
    x is N x d, d-dimensional feature (i.e., unique node ID in the featureless setting)
    '''    
    # Number of filter taps
    K = len(weights)

    # Create list to store graph diffused signals
    zs = [x]
    
    # Loop over the number of filter taps / different degree of S
    for k in range(1, K):        
        # diffusion step, S^k*x
        x = torch.spmm(S, x) #torch.matmul(x, S) -- slow
        # append the S^k*x in the list z
        zs.append(x)
    
    # sum up
    out = [z @ weight for z, weight in zip(zs, weights)]
    out = torch.stack(out)
    y = torch.sum(out, axis=0)
    return y

class GraphFilter(torch.nn.Module):

    def __init__(self, Fin, Fout, K, normalize=True):

        super(GraphFilter, self).__init__()
        self.Fin = Fin 
        self.Fout = Fout
        self.K = K
        self.normalize = normalize
        self.weight = nn.ParameterList([nn.Parameter(torch.randn(self.Fin,self.Fout)) for k in range(self.K)])
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Fin * self.K)
        for elem in self.weight:
          elem.data.uniform_(-stdv, stdv)


    def forward(self, x, edge_index, edge_weight):
        N = x.shape[0]
        E = edge_index.shape[1]
        if edge_weight is None:
            edge_weight = torch.ones(E)
        S = torch.sparse_coo_tensor(edge_index, edge_weight, (N,N))

        if self.normalize:
            S = S/N

        return LSIGF(self.weight,S,x)


class GNN(torch.nn.Module):

    def __init__(self, GNNtype, Flist, MLPlist, softmax=False, Klist = None):

        super(GNN, self).__init__()
        self.type = GNNtype
        self.Flist = Flist
        self.L = len(Flist)
        self.MLPlist = MLPlist 
        self.Lmlp = len(MLPlist)
        self.softmax = softmax
        self.Klist = Klist

        self.layers = nn.ModuleList()
        self.MLPlayers = nn.ModuleList()

        for i in range(self.L-1):
            if self.type == 'sage':
                self.layers.append(SAGEConv(Flist[i],Flist[i+1]))
            elif self.type == 'gcn':
                self.layers.append(GCNConv(Flist[i],Flist[i+1]))
            elif self.type =='gnn':
                self.layers.append(GraphFilter(Flist[i],Flist[i+1],Klist[i]))

        for i in range(self.Lmlp-1):
            self.MLPlayers.append(nn.Linear(MLPlist[i],MLPlist[i+1]))
            
    def forward(self, data):

        y, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for i, layer in enumerate(self.layers):
            if self.type == 'gnn':
                y = layer(y, edge_index=edge_index, edge_weight=edge_weight)
            else:
                y = layer(y, edge_index=edge_index)
            y = F.relu(y)

        for i, layer in enumerate(self.MLPlayers):
            y = layer(y)
        
        if self.softmax == True:
            y = F.log_softmax(y, dim=1)

        return y