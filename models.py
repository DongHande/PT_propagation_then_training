import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp
import math
import random

class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP(nn.Module):#
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return torch.log_softmax(self.Linear2(x), dim=1)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)

    def forward(self, x, adj):
        x = torch.relu(self.Linear1(torch.matmul(adj, x)))
        h = self.Linear2(torch.matmul(adj, x))
        return torch.log_softmax(h, dim=-1)


class SGCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(SGCN, self).__init__()
        self.Linear = Linear(nfeat, nclass, dropout, bias=False)
        self.x = None

    def forward(self, x, adj): 
        if self.x == None: 
            self.x = torch.matmul(adj, torch.matmul(adj, x))
        return torch.log_softmax(self.Linear(self.x), dim=-1)

class APPNP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, alpha):
        super(APPNP, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=False)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=False)
        self.alpha = alpha
        self.K = K

    def forward(self, x, adj):
        x = torch.relu(self.Linear1(x))
        h0 = self.Linear2(x)
        h = h0
        for _ in range(self.K):
            h = (1 - self.alpha) * torch.matmul(adj, h) + self.alpha * h0
        return torch.log_softmax(h, dim=-1)

class PT(nn.Module): 
    def __init__(self, nfeat, nhid, nclass, dropout, epsilon, mode, K, alpha):
        # mode: 0-PTS, 1-PTS, 2-PTA
        super(PT, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=True)
        self.epsilon = epsilon
        self.mode = mode
        self.K = K 
        self.alpha = alpha
        self.number_class = nclass

    def forward(self, x): 
        x = torch.relu(self.Linear1(x))
        return self.Linear2(x)

    def loss_function(self, y_hat, y_soft, epoch = 0): 
        if self.training: 
            y_hat_con = torch.detach(torch.softmax(y_hat, dim=-1))
            exp = np.log(epoch / self.epsilon + 1)
            if self.mode == 2: 
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(y_soft, y_hat_con**exp))) / self.number_class  # PTA
            elif self.mode == 1:
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(y_soft, y_hat_con))) / self.number_class  # PTD
            else: 
                loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.number_class # PTS
        else: 
            loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), y_soft)) / self.number_class
        return loss

    def inference(self, h, adj): 
        y0 = torch.softmax(h, dim=-1) 
        y = y0
        for _ in range(self.K):
            y = (1 - self.alpha) * torch.matmul(adj, y) + self.alpha * y0
        return y
