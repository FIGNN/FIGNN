# coding=utf-8
from __future__ import print_function
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
# import torch
import pickle as pkl
import sys
import random
import networkx as nx
import os
import torch
from sklearn import preprocessing

# seed = 123
# random.seed(seed)
# torch.random.manual_seed(seed)
label_set = ["Agents","AI","DB","IR","ML","HCI"]
# import networkx as nx
import pickle

# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(int(len(y) * 0.1))
    idx_val = range(int(len(y) * 0.1), int(len(y) * (0.1 + 0.2)))



    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj, indices, values, shape = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
import os


def pad_non_zero(features):
    feature_coo = features.tocoo()
    row_col_data = [(i,j,k) for i,j,k in zip(feature_coo.row, feature_coo.col, feature_coo.data)]
    row_col_data = sorted(row_col_data, key=lambda x:x[0])

    non_zero_batch = []
    nonzero_values_batch = []
    for idx, i in enumerate(row_col_data):
        # plus 1, leave the 0 as the padded index

        nonzero_index = i[1] + 1
        row = i[0]
        if row >= len(non_zero_batch):
            non_zero_batch.append([nonzero_index])
            nonzero_values_batch.append([i[2]])
        else:
            non_zero_batch[-1].append(nonzero_index)
            nonzero_values_batch[-1].append(i[2])

    return non_zero_batch, nonzero_values_batch

def load_data2(dataset_source, supervised=True, normalized=False):
    if os.path.exists("./data/{}_unsup".format(dataset_source)) and supervised is False:
        # with open("./data/{}_unsup".format(dataset_source), 'rb') as f1:
        #     data = pickle.load(f1)
        data = torch.load("./data/{}_unsup".format(dataset_source))
        adj = torch.sparse.FloatTensor(data['adj']['indices'], data['adj']['values'], data['adj']['shape'])
        features = data['features']
        labels = data['labels']
        val_edges = data['val_edges']
        val_edges_false = data['val_edges_false']
        test_edges = data['test_edges']
        test_edges_false = data['test_edges_false']
        train_edges = data['train_edges']
        train_edges_false = data['train_edges_false']
        return adj, features, labels, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


    data = sio.loadmat("data/{}.mat".format(dataset_source))
    features = data["Attributes"]
    no_norm_adj = data["Network"]
    labels = data["Label"]

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)


    no_norm_adj = (no_norm_adj + sp.eye(no_norm_adj.shape[0]))
    non_zero_index, non_zero_value = pad_non_zero(features)


    if supervised:

        adj = normalize_adj(no_norm_adj)
        node_perm = np.random.permutation(labels.shape[0])
        num_train = int(0.1 * adj.shape[0])
        num_val = int(0.2 * adj.shape[0])
        idx_train = node_perm[:num_train]
        idx_val = node_perm[num_train:num_train + num_val]
        idx_test = node_perm[num_train + num_val:]
    else:
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            mask_test_edges(no_norm_adj)
        adj = normalize_adj(adj_train)
        # remove the diangal
        adj_label = torch.FloatTensor((no_norm_adj - sp.eye(no_norm_adj.shape[0])).todense()).view(-1, 1)

    if normalized:
        features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.LongTensor(np.where(labels)[1])
    adj, indices, values, shape = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to_dense()

    labels = labels if supervised else adj_label
    if supervised:
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        return adj, features, non_zero_index,non_zero_value, labels, idx_train, idx_val, idx_test
    else:
        torch.save({
                'adj': {
                    'indices':indices,
                    'values': values,
                    'shape':shape
                },
                'non_zero_index':non_zero_index,
                'non_zero_value':non_zero_value,
                'features': features,
                'labels': labels,
                'val_edges': val_edges,
                'val_edges_false':val_edges_false,
                'test_edges': test_edges,
                'test_edges_false': test_edges_false,
            'train_edges':train_edges,
            'train_edges_false': train_edges_false
            }, "./data/{}_unsup".format(dataset_source))
        return adj, features, non_zero_index, non_zero_value, labels, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false





def normalized_adj(adj):

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj):
    adj_normalized = normalized_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
    # return normalized_adj(adj + sp.eye(adj.shape[0]))

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1),dtype='float')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features


#
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape), indices, values, shape


import random
def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # train 85% with positive links
    # val 5% positive links
    # train/test/val negative links have the same number of positive links respectively.
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    print("generate test edges")
    adj_dense = adj.todense()
    print("to dense finish")
    # no connection nodes
    adj_false = np.where(adj_dense == 0)
    edge_false = np.array([[i, j] for i, j in zip(adj_false[0], adj_false[1])])

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)


    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # n * n matrix
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))


    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    # make sure sample false edge equal to the real edge
    test_edges_false_idx = set(list(set(random.randint(0, len(edge_false)) for _ in range(2 * len(test_edges))))
                               [:len(test_edges)])
    val_test_false_idx = list(set(range(0, len(edge_false))) - test_edges_false_idx)


    val_edge_false_idx_new = list(set([val_test_false_idx[random.randint(0, len(val_test_false_idx))]
                                       for _ in range(2 * len(val_edges))]))[:len(val_edges)]

    train_edge_false_idx = list(set(range(0, len(edge_false))) - set(val_edge_false_idx_new) - test_edges_false_idx)
    train_edge_false_idx_new = list(set([train_edge_false_idx[random.randint(0, len(train_edge_false_idx))]
                                       for _ in range(2 * len(train_edges))]))[:len(train_edges)]

    test_edges_false_idx = list(test_edges_false_idx)



    test_edges_false = edge_false[test_edges_false_idx]
    val_edges_false = edge_false[val_edge_false_idx_new]
    train_edges_false = edge_false[train_edge_false_idx_new]




    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    print("finish edege mask")
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


#### Metrics

import  torch
def accuracy_node_label(output, labels, return_dis=False, supervised=True):
    # preds = output.max(1)[1].type_as(labels).view(-1, 1)
    if supervised:
        preds = output.max(1)[1].type_as(labels)
    else:
        preds = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    correct = preds.eq(labels).double()
    correct = correct.sum()
    if return_dis:
        return 1, 1
    else:
        return correct / labels.shape[0], torch.sum(preds) / preds.shape[0]
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, f1_score,confusion_matrix

def f1_score_torch(output, labels):
    preds = output.max(1)[1].type_as(labels).detach().cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(y_true=labels, y_pred=preds,average='weighted')

def confusion_score(output, labels):
    preds = output.max(1)[1].type_as(labels).detach().cpu().numpy()
    labels = labels.cpu().numpy()
    tn, fp, fn, tp = confusion_matrix( labels, preds).ravel()
    print("tn {}, fp {}, fn {}, tp {}".format(tn, fp, fn, tp))




def roc_auc_score_torch(y_true, y_pre):
    y_true = y_true.detach().cpu().view(-1).numpy()
    y_pre = y_pre.detach().cpu().view(-1).numpy()
    auc_score = roc_auc_score(y_true, y_pre)
    ap = average_precision_score(y_true, y_pre)
    return auc_score, ap

