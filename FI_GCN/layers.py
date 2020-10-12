import math
import numpy as np
from torch.nn import Embedding
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
#
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        try:
            assert not torch.isnan(h).any()
        except:
            th = 1
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class FMLayer(Module):

    def __init__(self, in_features, k_embedding):
        '''
        :param in_features: 输入特征维数
        :param k:  单一特征embedding
        :param bias:
        '''

        super(FMLayer, self).__init__()
        self.in_features = in_features
        self.k_embedding = k_embedding
        self.embedding = Embedding(in_features+1, k_embedding, padding_idx=0)

        # self.weight = Parameter(torch.FloatTensor(in_features,k_embedding))
        # self.reset_parameters()
        self.init_embedding()

    def init_embedding(self):
        init.xavier_uniform_(self.embedding.weight)
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




class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.map_value = torch.nn.Parameter(torch.tensor(10.), requires_grad=True)
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

        fm_paris_feature = fm_paris_feature.reshape(fm_paris_feature.shape[0] * fm_paris_feature.shape[1], fm_paris_feature.shape[2])
        fm_paris_feature = fm_paris_feature.unsqueeze(2)
        att_score = torch.bmm(gnn_feature_expand, fm_paris_feature)
        att_score = att_score.view(gnn_shape[0], gnn_shape[1], 1)
        att_score = torch.softmax(att_score, dim=1)
        return att_score

    def forward(self, gnn_feature, x, epoch):
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
        fm_pairs_feature = F.rel(self.attention(inner_product))
        attn_scores = self.interaction(fm_pairs_feature, gnn_feature)
        # attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)

        # attn_output = torch.sum(attn_scores * inner_product, dim=1) * inner_product.shape[1]
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        # attn_output = inner_product.shape[1] * torch.mean(inner_product, dim=1)
        # attn_output = F.dropout(attn_output, p=self.dropouts[1],  training=self.training)

        x_all = torch.cat((gnn_feature, attn_output), dim=1)
        return x_all





# class PersonalizedAttention(torch.nn.Module):
#     def __init__(self, nhid, attn_size, act="relu"):
#         super(PersonalizedAttention, self).__init__()
#         self.attention = torch.nn.Linear(nhid, attn_size)
#         self.projection = torch.nn.Linear(attn_size, 1)
#         if act == "relu":
#             self.act = torch.nn.ReLU()
#         else:
#             self.act = torch.nn.Sigmoid()
#
#     def forward(self, gcn_feature, fm_feature):
#         h_attn = torch.exp(self.act(
#             self.projection(
#                 self.attention(gcn_feature)
#             )
#         ))
#
#         f_attn = torch.exp(self.act(
#             self.projection(
#                 self.attention(fm_feature)
#             )
#         ))
#
#         h_attn_score = h_attn / (h_attn + f_attn)
#         f_attn_score = f_attn / (h_attn + f_attn)
#         # z = torch.cat((h_attn_score * gcn_feature, f_attn_score * fm_feature), dim=-1)
#         z = h_attn_score * gcn_feature + f_attn_score * fm_feature
#         return z
#
#
#
#
#
#
#
#
# class BI_Intereaction(Module):
#
#     def __init__(self, in_features, k_embedding):
#         '''
#         :param in_features: 输入特征维数
#         :param k:  单一特征embedding
#         :param bias:
#         '''
#
#         super(BI_Intereaction, self).__init__()
#         self.in_features = in_features
#         self.k_embedding = k_embedding
#         self.embedding = Embedding(in_features+1, k_embedding, padding_idx=0)
#         # TODO: consider dropout in Personalized Attention
#         self.per_attn = PersonalizedAttention(k_embedding, attn_size=k_embedding)
#
#         # self.weight = Parameter(torch.FloatTensor(in_features,k_embedding))
#         # self.reset_parameters()
#         self.init_embedding()
#
#     def init_embedding(self):
#         init.xavier_uniform_(self.embedding.weight)
#         # print('embedding_init',self.embedding.weight)
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#
#     def fm_forward(self, non_zero_index, non_zero_weight):
#         feature_embed = self.embedding(non_zero_index)
#         non_zero_weight = non_zero_weight.unsqueeze(-1)
#         middle = feature_embed * non_zero_weight
#         middle = middle * non_zero_weight
#         square_of_sum = torch.sum(middle, dim=1) ** 2
#         sum_of_square = torch.sum(middle ** 2, dim=1)
#         return 0.5 * (square_of_sum - sum_of_square)
#
#     def forward(self, gcn_feature, non_zero_index, non_zero_weight):
#         # embed_weight = self.embedding.weight
#         # embed_weight = embed_weight.unsqueeze(0)
#         # input = input.unsqueeze(-1)
#         # middle = torch.mul(embed_weight, input)
#         #
#         # left = torch.sum(middle, dim=1) ** 2
#         # right = torch.sum(middle ** 2, dim=1)
#         #
#         # output = 0.5 * (left - right)
#
#         fm_feature = self.fm_forward(non_zero_index, non_zero_weight)
#         output = self.per_attn(gcn_feature, fm_feature)
#
#
#         # output = self.atten_fm(middle)
#
#         return output
#
#     def attn_bi_pooling(self, input):
#         embed_weight = self.embedding.weight
#         embed_weight = embed_weight.unsqueeze(0)
#         input = input.unsqueeze(-1)
#         middle = torch.mul(embed_weight, input)
#         middle = self.atten_fm(middle)
#         return middle
#
#     #
#     # def forward(self, input):
#     #     return self.bi_pooling(input)







