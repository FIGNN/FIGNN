from dgl import DGLGraph
import scipy.sparse as sp
import scipy.io as sio
from sklearn import preprocessing
import numpy as np
import torch
import random
def pad_non_zero(features):
    feature_coo = features.tocoo()
    row_col_data = [(i,j,k) for i,j,k in zip(feature_coo.row, feature_coo.col, feature_coo.data)]
    row_col_data = sorted(row_col_data, key=lambda x:x[0])

    pad_length = 300
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

    non_zero_batch = [i[:pad_length] + [0] * (pad_length - len(i)) for i in non_zero_batch]
    nonzero_values_batch = [i[:pad_length] + [0] * (pad_length - len(i)) for i in nonzero_values_batch]
    return np.array(non_zero_batch), np.array(nonzero_values_batch)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_data(dataset_source):
    print("Load Dataset {}".format(dataset_source))
    data = sio.loadmat("./data/{}.mat".format(dataset_source))
    features = data["Attributes"]
    adj = data["Network"]
    labels = data["Label"]
    non_zero_index, non_zero_value = pad_non_zero(features)
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    # graphSage do not need self loop
    # adj = adj - sp.eye(adj.shape[0])

    node_perm = np.random.permutation(labels.shape[0])
    num_train = int(0.1 * adj.shape[0])
    num_val = int(0.2 * adj.shape[0])
    idx_train = node_perm[:num_train]
    idx_val = node_perm[num_train:num_train + num_val]
    idx_test = node_perm[num_train + num_val:]
    node_idx = list(range(adj.shape[0]))


    train_mask = [True if index in idx_train else False for index in range(adj.shape[0])]
    test_mask = [True if index in idx_test else False for index in range(adj.shape[0])]
    val_mask = [True if index in idx_val else False for index in range(adj.shape[0])]

    features = np.array(features.todense())
    labels = np.where(labels)[1]
    return adj, features, labels,node_idx, idx_train, idx_test, idx_val, train_mask, val_mask, test_mask, non_zero_index, non_zero_value,



def load_data2(dataset_source, supervised=True):
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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape), indices, values, shape

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

class NodeClfDataset(object):
    def __init__(self, dataset_source, self_loop=False):
        adj, features, labels, node_idx, idx_train, idx_test, idx_val, \
        train_mask, val_mask, test_mask, non_zero_index, non_zero_value = load_data(dataset_source)
        self.graph = DGLGraph(adj.tocoo(), readonly=True)
        self.features = features
        self.labels = labels
        self.num_labels = int(np.max(labels)) + 1
        # tarin/val/test indices
        self.train_mask = np.array(train_mask)
        self.val_mask = np.array(val_mask)
        self.test_mask = np.array(test_mask)
        self.non_zero_index = non_zero_index
        self.non_zero_value = non_zero_value

        print('Finished data loading.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def __getitem__(self, idx):
        assert idx == 0, "Reddit Dataset only has one graph"
        g = self.graph
        g.ndata['train_mask'] = self.train_mask
        g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['feat'] = self.features
        g.ndata['label'] = self.labels
        g.ndata['non_zero_index'] = self.non_zero_index
        g.ndata['non_zero_value'] = self.non_zero_value
        return g

    def __len__(self):
        return 1

class LinkPreDataset(object):
    def __init__(self, dataset_source, self_loop=False):
        adj, features, non_zero_index, non_zero_value, labels, train_edges, \
        train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = load_data2(dataset_source, supervised=False)
        self.graph = DGLGraph(adj.tocoo(), readonly=True)
        self.features = features
        self.labels = np.array([1] * len(train_edges) + [0] * len(train_edges_false) \
                      + [1] * len(test_edges) + [0] * len(test_edges_false))
        self.num_labels = 2
        # tarin/val/test indices
        self.train_mask = (train_edges, train_edges_false)
        self.val_mask = np.array(val_edges, val_edges_false)
        self.test_mask = np.array(test_edges, test_edges_false)
        self.non_zero_index = non_zero_index
        self.non_zero_value = non_zero_value

        print('Finished data loading.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))