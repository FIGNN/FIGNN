"""
This code was modified from the GCN implementation in DGL examples.
Simplifying Graph Convolutional Networks
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
SGC implementation in DGL.
"""
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.data import register_data_args

from dgl.nn.pytorch.conv import SGConv
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm



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
        fm_pairs_feature = F.relu(self.attention(inner_product))
        attn_scores = self.interaction(fm_pairs_feature, gnn_feature)
        attn_output = torch.sum(attn_scores * inner_product, dim=1) * 100




        x_all = torch.cat((gnn_feature, attn_output), dim=1)
        return x_all

class FMLayer(nn.Module):

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


class SimpleGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes, direct=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.model = SGConv(in_feats,
                       n_hidden,
                       k=2,
                       cached=True,
                       bias=args.bias)
        # self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        # self.dropout = nn.Dropout(dropout)
        # self.activation = activation
        self.fm = FMLayer(in_feats, n_hidden)
        self.pa_attn = AttentionalFactorizationMachine(n_hidden, n_hidden, dropouts=[0.2, 0.2])
        self.final_linear = nn.Linear(2 * n_hidden, n_classes)
        self.final_linear_single = nn.Linear(n_hidden, n_classes)
        self.direct = direct


    def forward(self, g, x, nonzer_index, nonzer_value):
        h = self.model(g, x)
        # bmp
        if self.direct:
            return self.final_linear_single(h)

        x_right = self.fm(nonzer_index, nonzer_value)
        x_all = self.pa_attn(h, x_right)
        x_all = self.final_linear(x_all)
        # x_all = self.final_linear_single(x_right)
        return x_all

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE mod    el on full neighbors (i.e. without neighbor sampling).
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
        nodes = torch.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

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


def evaluate(model, g, features, non_zero_index, non_zero_value, labels, mask, args, t1):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, non_zero_index, non_zero_value)[mask] # only compute the evaluation set
        labels = labels[mask]
        # logits = model(g, features, non_zero_index, non_zero_value) # only compute the evaluation set
        # labels = labels
        _, indices = torch.max(logits, dim=1)
        f1 = f1_score(y_pred=indices.cpu().numpy(), y_true=labels.cpu().numpy(), average="weighted")
        c_m = confusion_matrix(y_pred=indices.cpu().numpy(), y_true=labels.cpu().numpy())
        # print("MASK Length {}, True Count: {}, Type:{}".format(len(mask), sum(mask), type(mask)))
        # with open(f"./{args.dataset}_hello_{args.seed}.result", 'a') as f1:
        #     f1.write("")
            # f1.write(f"{type}-label:" + ",".join([str(i) for i in labels.cpu().numpy().tolist()]) + "\n")
            # f1.write(f"{type}-logits:" + ",".join([str(i) for i in indices.cpu().numpy().tolist()]) + "\n")
            # f1.write(f"{type}-all:" + ",".join([str(i) for i in raw_labels]) + "\n")

            # f1.write("Val:" + ",".join([str(i) for i in labels[val_mask]]) + "\n")
        # print(th.cpu().values().tolist())
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), f1

from dataset import OurDataset
import random
def main(args):
    output_f1 = open(args.output_file, 'w+')
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)

    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.gpu != -1:
        torch.cuda.manual_seed(args.seed)

    data = OurDataset(args.dataset,args.truncate_size, seed=args.seed)

    non_zero_index = data.non_zero_index
    non_zero_value = data.non_zero_value


    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    train_mask = train_mask.tolist()
    val_mask = val_mask.tolist()
    test_mask = test_mask.tolist()

    features = torch.Tensor(data.features)
    in_feats = features.shape[1]
    labels = torch.LongTensor(data.labels)
    n_classes = data.num_labels
    n_edges = g.number_of_edges()
    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features

    g.ndata['non_zero_index'] = non_zero_index
    g.ndata['non_zero_value'] = non_zero_value


    # create SGC model
    n_hidden = 32
    model = SimpleGCN(
        in_feats,
        n_hidden,
        n_classes, direct=args.direct

    )

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    # non_zero_index = torch.tensor(non_zero_index)
    # non_zero_value = torch.FloatTensor(non_zero_value)

    if cuda:
        non_zero_index = non_zero_index.cuda()
        non_zero_value = non_zero_value.cuda()
    dur = []
    acc =  0
    f1 = 0
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features, non_zero_index, non_zero_value) # only compute the train set
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc_here, f1_here = evaluate(model, g, features, non_zero_index, non_zero_value, labels, test_mask, args, 'val')
        if acc_here > acc:
            acc = acc_here
            f1 = f1_here
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | F1 {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc,f1, n_edges / np.mean(dur) / 1000))

    # model, g, features, non_zero_index, non_zero_value, labels, mask
    # acc, f1 = evaluate(model, g, features, non_zero_index, non_zero_value, labels, test_mask, args, 'test')
    print("Test Accuracy {:.4f}, F1 {:.4f}".format(acc,f1))
    output_f1.write("Acc: {:.4f}, F1: {:.4f}".format(acc, f1))
    output_f1.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGC')
    # register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--seed", type=int, default=123,
            help="seed")
    parser.add_argument("--lr", type=float, default=0.2,
            help="learning rate")
    parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
    parser.add_argument("--n_epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-6,
            help="Weight for L2 loss")
    parser.add_argument("--dataset", type=str, default="BlogCatalog")
    parser.add_argument("--output_file", type=str, default="./simple_gcn.result")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--truncate_size", type=int, default=200)
    args = parser.parse_args()
    print(args)

    main(args)
