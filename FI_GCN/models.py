import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, FMLayer, PersonalizedAttention, GraphAttentionLayer, SpGraphAttentionLayer, AttentionalFactorizationMachine
import torch
import dgl
import dgl.nn.pytorch as dglnn

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, supervised=True,  direct=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)

        self.fm = FMLayer(nfeat, nhid)
        if supervised:
            n_output = nclass
        else:
            n_output = nhid
        self.final_linear = nn.Linear(2 * nhid, n_output)
        self.final_linear_single = nn.Linear(nhid, n_output)
        # self.final_linear = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.supervised = supervised
        self.pa_attn = AttentionalFactorizationMachine(nhid, nhid, dropouts=[0.2, 0.2])
        self.direct = direct
        # self.pa_attn = PersonalizedAttention(nhid, nhid)


    def forward(self, x, adj, type_index, non_zero_index, non_zero_value, epoch=None,):

        x_left = F.relu(self.gc1(x, adj))
        x_left = self.dropout(x_left)
        x_left = self.gc2(x_left, adj)
        if type_index == None:
            type_index = list(range(x_left))
        if self.direct:
            x_all = self.final_linear_single(x_left[type_index])
        else:
            x_right = self.fm(non_zero_index, non_zero_value)
            x_all = self.pa_attn(x_left[type_index], x_right, epoch)
            x_all = self.final_linear(x_all)

        if self.supervised:
            return F.log_softmax(x_all, dim=1)
        else:
            return x_all

class SpGAT(nn.Module):
    # nfeat, nhid, nclass, dropout, alpha, nheads
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, direct=False, supervised=True):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)
        self.fm = FMLayer(nfeat, nhid)
        self.supervised = supervised
        if self.supervised:
            n_output = nclass
        else:
            n_output = nhid

        self.final_linear = nn.Linear(2 * nhid, n_output)
        self.final_linear_single = nn.Linear(nhid, n_output)
        # self.final_linear = nn.Linear(nhid, nclass)
        self.pa_attn = AttentionalFactorizationMachine(nhid, nhid, dropouts=[0.2, 0.2])
        self.direct = direct


    def forward(self, x, adj, type_index, non_zero_index, non_zero_value, epoch=None):
        x1 = F.dropout(x, self.dropout, training=self.training)
        x1 = torch.cat([att(x1, adj) for att in self.attentions], dim=1)

        x1 = F.dropout(x1, self.dropout, training=self.training)
        gat_feature = F.elu(self.out_att(x1, adj))
        if self.direct:
            x_all = self.final_linear_single(gat_feature[type_index])
        else:
            fm_feature = self.fm(non_zero_index, non_zero_value)
            x_all = self.pa_attn(gat_feature[type_index], fm_feature, epoch)
            x_all = self.final_linear(x_all)
        if self.supervised:
            return F.log_softmax(x_all, dim=1)
        else:
            return x_all
