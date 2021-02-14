import arguments
import numpy as np

import torch
import torch.nn.functional as F

from utils import accuracy, load_data
from models import APPNP
import random
from early_stop import EarlyStopping, Stop_args


args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.ini_seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(args.ini_seed)

# Load data and pre_process data 
adj, features, labels, idx_train, idx_val, idx_test = load_data(graph_name = args.dataset, str_noise_rate=args.str_noise_rate, seed = args.seed)

# Model and optimizer
model = APPNP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, 
            K=args.K, 
            alpha=args.alpha).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                       lr=args.lr)

features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + args.weight_decay * torch.sum(model.Linear1.weight ** 2) / 2
    loss_train.backward()
    optimizer.step()

    acc_train = accuracy(output[idx_train], labels[idx_train])

    # Evaluate validation set performance separately,
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_val: {:.4f}'.format(acc_val.item()))

    return loss_val.item(), acc_val.item()

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)
for epoch in range(args.epochs):
    loss_val, acc_val = train(epoch)
    if early_stopping.check([acc_val, loss_val], epoch):
        break

print("Optimization Finished!")

# Restore best model
print('Loading {}th epoch'.format(early_stopping.best_epoch))
model.load_state_dict(early_stopping.best_state)
test()

