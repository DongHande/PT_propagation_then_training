import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random
import data.io as io
import copy

def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719):
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx

def exclude_idx(idx: np.ndarray, idx_exclude_list):
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])

def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20,
        nstopping: int = 500, seed: int = 2413340114):
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(
                idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
            exclude_idx(idx, [train_idx]),
            nstopping, replace=False)
    return train_idx, stopping_idx

def gen_splits(labels: np.ndarray, idx_split_args,
        test: bool = False):
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
            all_idx, idx_split_args['nknown'])
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, stopping_idx = train_stopping_split(
            known_idx, labels[known_idx], **stopping_split_args)
    if test:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    return train_idx, stopping_idx, val_idx

def load_data(graph_name = 'cora_ml', lbl_noise=0, str_noise_rate=0.1, seed = 2144199730):
    dataset = io.load_dataset(graph_name)
    dataset.standardize(select_lcc=True)
    features = dataset.attr_matrix
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    # features = sparse_mx_to_torch_sparse_tensor(features)
    labels = dataset.labels
    adj = dataset.adj_matrix
    adj = str_noise(adj, labels, str_noise_rate, seed)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if graph_name == 'ms_academic':
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 5000, 'seed': seed}
    else:
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': seed}
    idx_train, idx_val, idx_test = gen_splits(labels, idx_split_args, test=True)

    labels = add_label_noise(idx_train, labels, lbl_noise, seed)
    
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def label_propagation(adj, labels, idx, K, alpha): 
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for i in idx:
        y0[i][labels[i]] = 1.0
    
    y = y0
    for i in range(K): 
        y = torch.matmul(adj, y)
        for i in idx:
            y[i] = F.one_hot(torch.tensor(labels[i].cpu().numpy().astype(np.int64)), labels.max().item() + 1)
        y = (1 - alpha) * y + alpha * y0
    return y
    
def add_label_noise(idx_train, labels, noise_num, seed): 
    if noise_num == None: 
        return labels
    random.seed(seed)
    for c in range(max(labels)+1): 
        sele_idx = [i + c * 20 for i in random.sample(range(20), noise_num)]
        for i in sele_idx: 
            re_lb = random.sample([i for i in range(max(labels)+1) if i != c], 1)
            labels[idx_train[i]] = re_lb[0]
    return labels

def str_noise(adj, labels, noise_rate, seed = 0):
    if noise_rate > 1.0:
        return adj
    idx = np.arange(len(labels))
    adj = adj.tocoo().astype(np.float32)

    row = adj.row
    col = adj.col

    upper_edge = np.arange(len(row))[row < col]
    idx_upper_edge = np.arange(len(upper_edge))
    good_edge_idx = idx_upper_edge[labels[row[upper_edge]] == labels[col[upper_edge]]]
    bad_edge_idx = idx_upper_edge[labels[row[upper_edge]] != labels[col[upper_edge]]]

    origin_noise_rate = len(bad_edge_idx) / len(idx_upper_edge)

    rnd_state = np.random.RandomState(seed)
    random.seed(seed)
    if origin_noise_rate > noise_rate: 
        sub_num = int(len(upper_edge) * (origin_noise_rate - noise_rate))
        inv_idx = rnd_state.choice(bad_edge_idx, sub_num, replace=False)
        for i in inv_idx: 
            row_i = row[[upper_edge[i]]]
            col_i = col[[upper_edge[i]]]
            if random.random() > 0.5: 
                lbl = labels[row_i]
                col_new = rnd_state.choice(idx[labels == lbl], 2, replace=False)
                if col_new[0] != row_i: 
                    col_new = col_new[0]
                else: 
                    col_new = col_new[1]
                col[[upper_edge[i]]] = col_new
                for j in range(len(row)): 
                    if row[j] == col_i and col[j] == row_i: 
                        row[j] = col_new
                        break
            else: 
                lbl = labels[col_i]
                row_new = rnd_state.choice(idx[labels == lbl], 2, replace=False)
                if row_new[0] != col_i: 
                    row_new = row_new[0]
                else: 
                    row_new = row_new[1]
                row[[upper_edge[i]]] = row_new
                for j in range(len(row)): 
                    if row[j] == col_i and col[j] == row_i: 
                        col[j] = row_new
                        break
    else: 
        add_num = int(len(upper_edge) * (noise_rate - origin_noise_rate))
        inv_idx = rnd_state.choice(good_edge_idx, add_num, replace=False)
        for i in inv_idx: 
            row_i = row[[upper_edge[i]]]
            col_i = col[[upper_edge[i]]]
            if random.random() > 0.5: 
                lbl = labels[row_i]
                col_new = rnd_state.choice(idx[labels != lbl], 1, replace=False)
                col_new = col_new[0]
                col[[upper_edge[i]]] = col_new
                for j in range(len(row)): 
                    if row[j] == col_i and col[j] == row_i: 
                        row[j] = col_new
                        break
            else: 
                lbl = labels[col_i]
                row_new = rnd_state.choice(idx[labels != lbl], 1, replace=False)
                row_new = row_new[0]
                row[[upper_edge[i]]] = row_new
                for j in range(len(row)): 
                    if row[j] == col_i and col[j] == row_i: 
                        col[j] = row_new
                        break

    adj.row = row
    adj.col = col
    return adj

def get_noise_rate(adj, labels):
    indices = adj._indices().numpy()
    upper = indices[0, :] > indices[1, :]
    upper_indices = indices[:, upper]

    bad_num = 0
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() != labels[j].item():
            bad_num += 1

    return bad_num / upper_indices.shape[1]