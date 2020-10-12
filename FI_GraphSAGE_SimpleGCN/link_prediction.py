import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--cuda', action='store_true',
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--truncate_size', type=int, default=200,
                    help='Feature size for fm')


parser.add_argument('--output_file', type=str, default="./delete.result",
                    help='output file location')


parser.add_argument('--data_name', type=str, default="BlogCatalog",
                    help='Dataset Name', choices=['BlogCatalog','Flickr','ACM','DBLP'])

parser.add_argument('--direct', action="store_true",)
parser.add_argument('--model_type', default="graphsage",)

args = parser.parse_args()

output_file = open(args.output_file, 'w')
# Load Pytorch as backend
dgl.load_backend('pytorch')
import numpy as np

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

from dgl.nn.pytorch import conv as dgl_conv
from graphSage import AttentionalFactorizationMachine, FMLayer
from dgl.nn.pytorch.conv import SGConv

class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,direct=False):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type,
                                         feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type,
                                             feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type,
                                         feat_drop=dropout, activation=None))

        self.fm = FMLayer(in_feats, n_hidden)
        self.pa_attn = AttentionalFactorizationMachine(n_hidden, n_hidden, dropouts=[0.2, 0.2])
        self.final_linear = nn.Linear(2 * n_hidden, out_dim)
        self.final_linear_single = nn.Linear(n_hidden, out_dim)
        self.direct = direct

    def forward(self, g, features, nonzer_index, nonzer_value):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        if self.direct:
            return self.final_linear_single(h)
        x_right = self.fm(nonzer_index, nonzer_value)
        x_all = self.pa_attn(h, x_right)
        x_all = self.final_linear(x_all)

        return x_all

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
                       bias=False)
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


from dataset import load_data2
max_length_all = 100
adj, features, non_zero_index, non_zero_value, labels, train_edges, train_edges_false, val_edges, \
val_edges_false, test_edges, test_edges_false = load_data2(args.data_name, seed=args.seed, truncate_size=max_length_all, supervised=False)
# load and preprocess the pubmed dataset
# non_zero_index = torch.tensor([i[:max_length_all] + [0] * (max_length_all - len(i)) for i in non_zero_index])
# non_zero_value = torch.FloatTensor([i[:max_length_all] + [0] * (max_length_all - len(i)) for i in non_zero_value])
non_zero_index = torch.tensor(non_zero_index)
non_zero_value = torch.FloatTensor(non_zero_value)
# not utilize the validation
val_edges_false = test_edges_false
val_edges = test_edges

g = DGLGraph(adj.tocoo(), readonly=True)

in_feats = features.shape[1]
#Model hyperparameters
n_hidden = args.hidden
n_layers = 2
dropout = 0.5
aggregator_type = 'mean'

# create GraphSAGE model
if args.model_type == "graphsage":
    gconv_model = GraphSAGEModel(in_feats,
                                 n_hidden,
                                 n_hidden,
                                 n_layers,
                                 F.relu,
                                 dropout,
                                 aggregator_type, args.direct)

else:
    gconv_model = SimpleGCN(
        in_feats, n_hidden, 2, direct=args.direct
    )


# NCE loss
def NCE_loss(pos_score, neg_score, neg_sample_size):
    pos_score = F.logsigmoid(pos_score)
    neg_score = F.logsigmoid(-neg_score)
    return -pos_score - neg_score

class LinkPrediction(nn.Module):
    def __init__(self, gconv_model):
        super(LinkPrediction, self).__init__()
        self.gconv_model = gconv_model

    def forward(self, g, features, positive_edges, negative_edges, non_zero_index, non_zero_value):
        emb = self.gconv_model(g, features, non_zero_index, non_zero_value)
        pos_score = score_func(positive_edges, emb)
        neg_score = score_func(negative_edges, emb)
        return torch.mean(NCE_loss(pos_score, neg_score, neg_sample_size))


def score_func(edges, embeddings):
    node_0 = edges[:, 0]
    node_1 = edges[:, 1]
    value = []
    for i in range(0, len(node_0), 1000):
        node_0_slice = node_0[i:i+1000]
        node_1_slice = node_1[i:i+1000]
        node_0_e = embeddings[node_0_slice]
        node_1_e = embeddings[node_1_slice]
        value.append(torch.sum(node_0_e * node_1_e, dim=1))
    value = torch.cat(tuple(value), dim=0)
    return value

# def score_func(g, emb):
#
#     # Read the node embeddings of the source nodes and destination nodes.
#     pos_heads = emb[src_nid]
#     pos_tails = emb[dst_nid]
#     return torch.sum(pos_heads * pos_tails, dim=1)

from sklearn.metrics import roc_auc_score, average_precision_score
def metric_func(y_pred, y_label):
    y_pred = torch.sigmoid(y_pred)
    preds = torch.where(y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred))
    correct = preds.eq(y_label.float()).double()
    correct = correct.sum() / len(y_label)

    y_true = y_label.detach().cpu().view(-1).numpy()
    y_pre = y_pred.detach().cpu().view(-1).numpy()
    auc_score = roc_auc_score(y_true, y_pre)
    ap = average_precision_score(y_true, y_pre)
    
    return correct, auc_score, ap

def LPEvaluate(gconv_model, g, features, eval_eids, neg_sample_size, positive_edges, negative_edges,non_zero_index, non_zero_value):
    gconv_model.eval()
    with torch.no_grad():
        emb = gconv_model(g, features,non_zero_index, non_zero_value)
        
        # pos_g, neg_g = edge_sampler(g, neg_sample_size, eval_eids, return_false_neg=True)
        pos_score = score_func(positive_edges, emb)
        neg_score = score_func(negative_edges, emb)
        return metric_func(torch.cat([pos_score, neg_score]), torch.tensor([1] * len(positive_edges) + [0] * len(negative_edges)))
    

eids = np.random.permutation(g.number_of_edges())
train_eids = eids[:int(len(eids) * 0.8)]
valid_eids = eids[int(len(eids) * 0.8):int(len(eids) * 0.9)]
test_eids = eids[int(len(eids) * 0.9):]
train_g = g.edge_subgraph(train_eids, preserve_nodes=True)



# Model for link prediction
model = LinkPrediction(gconv_model)

# Training hyperparameters
weight_decay = 5e-4
n_epochs = args.epochs
lr = args.lr
neg_sample_size = 100

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initialize graph
dur = []
for epoch in range(n_epochs):
    model.train()
    loss = model(g, features, train_edges, train_edges_false, non_zero_index, non_zero_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        acc, auc_score, ap = LPEvaluate(gconv_model, g, features, valid_eids, neg_sample_size, test_edges, test_edges_false, non_zero_index, non_zero_value)
        print("Epoch {:05d} | Loss {:.4f} | MRR {:.4f}  | AUC {:.4f} | AP {:.4f}"
          .format(epoch, loss.item(), acc, auc_score, ap))
        output_file.write("Epoch {:05d} | Loss {:.4f} | MRR {:.4f}  | AUC {:.4f} | AP {:.4f}\n"
          .format(epoch, loss.item(), acc, auc_score, ap))
        

print()
# Let's save the trained node embeddings.
acc, auc_score, ap = LPEvaluate(gconv_model, g, features, valid_eids, neg_sample_size, test_edges, test_edges_false, non_zero_index, non_zero_value)
print("Final Test: MRR {:.4f} | AUC {:.4f} | AP {:.4f}"
          .format(acc, auc_score, ap))
output_file.write("Final Test: MRR {:.4f} | AUC {:.4f} | AP {:.4f}"
          .format(acc, auc_score, ap))

output_file.close()


