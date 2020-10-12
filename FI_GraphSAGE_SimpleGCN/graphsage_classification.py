import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
import tqdm
import traceback
from dataset import OurDataset
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import math

#### Neighbor sampler

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def interaction(self, fm_paris_feature, gnn_feature):
        gnn_feature_expand = gnn_feature.unsqueeze(1)
        gnn_feature_expand = gnn_feature_expand.unsqueeze(2)
        feature_pair_count = fm_paris_feature.shape[1]
        gnn_feature_expand = gnn_feature_expand.expand(-1, feature_pair_count, -1, -1)
        # interaction
        gnn_shape = gnn_feature_expand.shape
        gnn_feature_expand = gnn_feature_expand.reshape(gnn_shape[0] * gnn_shape[1],
                                                     gnn_shape[2], gnn_shape[3])
        fm_paris_feature = fm_paris_feature.reshape(fm_paris_feature.shape[0] * fm_paris_feature.shape[1],fm_paris_feature.shape[2])
        fm_paris_feature = fm_paris_feature.unsqueeze(2)
        att_score = torch.bmm(gnn_feature_expand, fm_paris_feature)
        att_score = att_score.view(gnn_shape[0], gnn_shape[1], 1)
        att_score = torch.softmax(att_score, dim=1)
        return att_score




    def forward(self, gnn_feature, x ):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q

        # attn_scores = F.relu(self.attention(inner_product))
        # attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        fm_pairs_feature = F.relu(self.attention(inner_product))
        attn_scores = self.interaction(fm_pairs_feature, gnn_feature)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)
        attn_output = torch.sum(attn_scores * inner_product, dim=1) * inner_product.shape[1]
        attn_output = F.dropout(attn_output, p=self.dropouts[1],  training=self.training)

        x_all = torch.cat((gnn_feature, attn_output), dim=1)
        return x_all

class FMLayer(torch.nn.Module):

    def __init__(self, in_features, k_embedding):
        '''
        :param in_features: 输入特征维数
        :param k:  单一特征embedding
        :param bias:
        '''

        super(FMLayer, self).__init__()
        self.in_features = in_features
        self.k_embedding = k_embedding
        self.embedding = nn.Embedding(in_features+1, k_embedding, padding_idx=0)

        # self.weight = Parameter(torch.FloatTensor(in_features,k_embedding))
        # self.reset_parameters()
        self.init_embedding()

    def init_embedding(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        # print('embedding_init',self.embedding.weight)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, nonzero_index, nonzero_value):
        feature_embed = self.embedding(nonzero_index)
        nonzero_value = nonzero_value.unsqueeze(-1)
        feature_vector = feature_embed * nonzero_value

        return feature_vector
        # left = torch.sum(feature_vector, dim=1) ** 2
        # right = torch.sum(feature_vector ** 2, dim=1)
        #
        # output = 0.5 * (left - right)

        # return output


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout, direct=False):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        agg_type = "mean"
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, agg_type))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, agg_type))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, agg_type))
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fm = FMLayer(in_feats, n_hidden)
        self.pa_attn = AttentionalFactorizationMachine(n_hidden, n_hidden, dropouts=[0.2, 0.2])
        self.final_linear = nn.Linear(2 * n_hidden, n_classes)
        self.final_linear_single = nn.Linear(n_hidden, n_classes)
        self.direct = direct


    def forward(self, blocks, x, nonzer_index, nonzer_value, ):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        # bmp
        if self.direct:
            return self.final_linear_single(h)
        x_right = self.fm(nonzer_index, nonzer_value)
        x_all = self.pa_attn(h, x_right)
        x_all = self.final_linear(x_all)
        return x_all

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y



        return y



def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.
    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def compute_f1(pred, labels):
    pred_list = th.argmax(pred, dim=1).tolist()
    label_list = labels.tolist()
    f1 = f1_score(label_list, pred_list, average="weighted")
    return f1


def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask]), compute_f1(pred[val_mask], labels[val_mask])

def block_eval(eval_dataloader, model, fout):
    batch_pred_list = []
    batch_labels_list  = []
    model.eval()
    with torch.no_grad():
        for step, blocks in enumerate(eval_dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            batch_inputs, nonzer_index, nonzer_value, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs, nonzer_index, nonzer_value)
            batch_pred_list.append(batch_pred)
            batch_labels_list.append(batch_labels)
        acc = compute_acc(torch.cat(batch_pred_list, dim=0), torch.cat(batch_labels_list, dim=0))
        f1 = compute_f1(torch.cat(batch_pred_list, dim=0), torch.cat(batch_labels_list, dim=0))
        print("Eval ACC: {}, F1: {}".format(acc.item(), f1))
        fout.write("Eval ACC: {}, F1: {}\n".format(acc.item(), f1))

def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    nonzer_index = torch.tensor(g.ndata['non_zero_index'][seeds]).to(device)
    nonzer_value = torch.FloatTensor(g.ndata['non_zero_value'][seeds]).to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, nonzer_index, nonzer_value, batch_labels


#### Entry point
def run(args, device, data, output_f1):
    # Unpack data
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = th.LongTensor(np.nonzero(train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    train_dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    eval_dataloader = DataLoader(
        dataset=val_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.direct)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()
        model.train()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, blocks in enumerate(train_dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            batch_inputs, nonzer_index, nonzer_value, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs, nonzer_index, nonzer_value)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                f1 = compute_f1(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} F1 {:4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), acc.item(), f1, np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        # print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            block_eval(eval_dataloader, model, output_f1)
            # eval_acc, eval_f1 = evaluate(model, g, g.ndata['features'], labels, val_mask, args.batch_size, device)
            # print('Eval Acc {:.4f} F1 {:.4f}'.format(eval_acc, eval_f1))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=100)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--direct', action="store_true")
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--data_name', type=str, default="BlogCatalog",
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--output_file', type=str, default="result/result.output",
                           help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()
    print(args)
    output_f1 = open(args.output_file,'w+')
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load reddit data
    data = OurDataset(self_loop=True, dataset_source=args.data_name)
    train_mask = data.train_mask
    val_mask = data.test_mask
    features = th.Tensor(data.features)
    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels
    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features
    g.ndata['non_zero_index'] = data.non_zero_index
    g.ndata['non_zero_value'] = data.non_zero_value
    prepare_mp(g)
    # Pack data
    data = train_mask, val_mask, in_feats, labels, n_classes, g

    run(args, device, data, output_f1)
    output_f1.close()
