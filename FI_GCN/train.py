from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from util import *
from models import GCN, SpGAT
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
# Training settings
# Namespace(cuda=False, dropout=0.5, epochs=200, fastmode=False, hidden=32, lr=0.01, seed=42, supervised=True, weight_decay=0.0005)
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true',
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
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
parser.add_argument('--supervised', action='store_true',
                    help='whether supervised ')

parser.add_argument('--truncate_size', type=int, default=200,
                    help='Feature size for fm')

parser.add_argument('--num_heads', type=int, default=8,
                    help="number of attention heads")

parser.add_argument('--alpha', type=float,
                    help="Attention Alpha", default=0.2)


parser.add_argument('--output_file', type=str, default="./delete.result",
                    help='output file location')


parser.add_argument('--data_name', type=str, default="BlogCatalog",
                    help='Dataset Name', choices=['BlogCatalog','Flickr','ACM','DBLP'])


parser.add_argument('--model_type', type=str, default="gcn",
                    help="GNN model", choices=['gcn','gat','graphsage'])

parser.add_argument('--direct', action="store_true",)



args = parser.parse_args()
print(args)
print("Use Cuda is {}".format(args.cuda))

str_name = "supervised" if args.supervised else "unsupervised"

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

normalized = True if args.model_type == 'gat' else False
data_name = args.data_name
# Load data
# adj, features, labels, idx_train, idx_test,idx_val  = load_data(Content_path, Cites_path)
# adj, features, labels, idx_train, idx_test , idx_val = load_data_pubmed(data_name)
# adj, features, labels, idx_train, idx_test , idx_val  = load_dataset()
# adj, features, labels, idx_train, idx_val, idx_test = load_data2(data_name, supervised=args.supervised)
if args.supervised is False:
    adj, features, non_zero_index, non_zero_value, labels, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = load_data2(data_name, supervised=False)
else:
    adj, features, non_zero_index, non_zero_value, labels, idx_train, idx_val, idx_test = load_data2(data_name, supervised=True, normalized=normalized)

#### Truncate and Pad the sentence ####
train_non_zero_index = [non_zero_index[i] for i in idx_train.tolist()]
train_non_zero_value =[non_zero_value[i] for i in idx_train.tolist()]

val_non_zero_index = [non_zero_index[i] for i in idx_val.tolist()]
val_non_zero_value = [non_zero_value[i] for i in idx_val.tolist()]

test_non_zero_index = [non_zero_index[i] for i in idx_test.tolist()]
test_non_zero_value = [non_zero_value[i] for i in idx_test.tolist()]
# pad all the tokens
truncate_size = args.truncate_size
max_train_len = min(max([len(i) for i in train_non_zero_index]), truncate_size)
train_non_zero_index = torch.tensor([i[:max_train_len]+[0]*(max_train_len - len(i)) for i in train_non_zero_index])
train_non_zero_value = torch.FloatTensor([i[:max_train_len] +[0]*(max_train_len - len(i)) for i in train_non_zero_value])

max_test_len = min(max([len(i) for i in test_non_zero_index]), truncate_size)
test_non_zero_index = torch.tensor([i[:max_test_len] +[0]*(max_test_len - len(i)) for i in test_non_zero_index])
test_non_zero_value = torch.FloatTensor([i[:max_test_len] +[0]*(max_test_len - len(i)) for i in test_non_zero_value])

max_val_len = min(max([len(i) for i in val_non_zero_index]), truncate_size)
val_non_zero_index = torch.tensor([i[:max_val_len]+[0]*(max_val_len - len(i)) for i in val_non_zero_index])
val_non_zero_value = torch.FloatTensor([i[:max_val_len]+[0]*(max_val_len - len(i)) for i in val_non_zero_value])

max_length_all = min(max([len(i) for i in non_zero_value]), truncate_size)
non_zero_index = torch.tensor([i[:max_length_all] + [0] * (max_length_all - len(i)) for i in non_zero_value])
non_zero_value = torch.FloatTensor([i[:max_length_all] + [0] * (max_length_all - len(i)) for i in non_zero_value])

#### Truncate and Pad the sentence ####


print(features)
# print(labels[idx_train])
# print(adj)
# Model and optimizer
if args.model_type == 'gcn':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass= int(torch.max(labels).item()) + 1,
                dropout=args.dropout,
                supervised=args.supervised,
                direct=args.direct)
elif args.model_type == 'gat':
    model = SpGAT(
                nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(torch.max(labels).item()) + 1,
                dropout=args.dropout, alpha=args.alpha, nheads=args.num_heads, direct=args.direct,
                supervised=args.supervised)
else:
    raise NotImplementedError
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# this means use all the point
idx_all = None
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    train_non_zero_index = train_non_zero_index.cuda()
    train_non_zero_value = train_non_zero_value.cuda()

    val_non_zero_index = val_non_zero_index.cuda()
    val_non_zero_value = val_non_zero_value.cuda()

    test_non_zero_index = test_non_zero_index.cuda()
    test_non_zero_value = test_non_zero_value.cuda()

    non_zero_index = non_zero_index.cuda()
    non_zero_value = non_zero_value.cuda()

def edge_ratio(edges, embeddings):
    node_0 = edges[:, 0]
    node_1 = edges[:, 1]
    node_0_e = embeddings[node_0]
    node_1_e = embeddings[node_1]
    return torch.sigmoid(torch.sum(node_0_e * node_1_e, dim=1))

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    if args.supervised:
        output = model(features, adj, idx_train, train_non_zero_index, train_non_zero_value, epoch)
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train, train_ratio = accuracy_node_label(output, labels[idx_train], False, True)
    else:
        embeddings = model(features, adj, idx_all, non_zero_index, non_zero_value, epoch)
        pos_pre = edge_ratio(train_edges, embeddings)
        neg_pre = edge_ratio(train_edges_false, embeddings)
        adj_pre = torch.cat((pos_pre, neg_pre), dim=0)
        labels_here = torch.cat((torch.ones_like(pos_pre), torch.zeros_like(neg_pre)), dim=0)
        loss_fn = torch.nn.BCELoss()
        loss_train = loss_fn(adj_pre, labels_here)
        acc_train, train_ratio = accuracy_node_label(adj_pre, labels_here, False, False)


    loss_train.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), 5)
    optimizer.step()


    model.eval()

    if args.supervised:
        output = model(features, adj, idx_val, val_non_zero_index, val_non_zero_value)
        loss_val = F.nll_loss(output, labels[idx_val])
        acc_val, val_ratio = accuracy_node_label(output, labels[idx_val], False, True)
        f1_val = f1_score_torch(output, labels[idx_val])
    else:
        embeddings = model(features, adj, idx_all, non_zero_index, non_zero_value, epoch)
        pos_edge_pre = edge_ratio(val_edges, embeddings)
        neg_edge_pre = edge_ratio(val_edges_false, embeddings)
        loss_fn = torch.nn.BCELoss()
        link_pre = torch.cat((pos_edge_pre, neg_edge_pre), dim=0).view(-1, 1)

        link_labels = torch.cat((pos_edge_pre.new_ones(pos_edge_pre.shape[0], 1),
                                 neg_edge_pre.new_zeros(neg_edge_pre.shape[0], 1)), dim=0)
        loss_val = loss_fn(link_pre, link_labels)
        acc_val, val_ratio = accuracy_node_label(link_pre, link_labels, False, False)
        auc_val, ap_val = roc_auc_score_torch(link_labels, link_pre)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy_node_label(output[idx_val], labels[idx_val])
    if args.supervised:
        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'train_ratio: {:.4f}'.format(train_ratio.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'f1_val: {:.4f}'.format(f1_val),
          'val_ratio: {:.4f}'.format(val_ratio.item()),
          'time: {:.4f}s'.format(time.time() - t),
          'type :{}'.format(str_name)
          )
    else:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'train_ratio: {:.4f}'.format(train_ratio.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'val_ratio: {:.4f}'.format(val_ratio.item()),
              'auc: {}'.format(auc_val),
              'ap: {}'.format(ap_val),
              'time: {:.4f}s'.format(time.time() - t),
              'type :{}'.format(str_name)
              )
    if (epoch + 1) % 5 == 0:
        test()
import gc
from sklearn.metrics import accuracy_score, f1_score
def test():
    model.eval()

    if args.supervised:
        output_list = []
        batch = 1000
        for idx in range(0, len(idx_test), batch):
            type_idx = idx_test[idx:idx+batch]
            test_non_zero_index_slice = test_non_zero_index[idx:idx+batch]
            test_non_zero_value_slide = test_non_zero_value[idx:idx+batch]
            output = model(features, adj, type_idx, test_non_zero_index_slice, test_non_zero_value_slide)
            output_list.extend(torch.argmax(output, dim=1).tolist())
        loss_test = 0
        test_ratio = 0
        label_test = labels[idx_test].tolist()
        acc_test = accuracy_score(output_list, label_test)
        f1_test = f1_score(output_list, label_test, average='weighted')

    else:
        embeddings = model(features, adj, idx_all, non_zero_index, non_zero_value, epoch)
        pos_edge_pre = edge_ratio(val_edges, embeddings)
        neg_edge_pre = edge_ratio(val_edges_false, embeddings)
        loss_fn = torch.nn.BCELoss()
        link_pre = torch.cat((pos_edge_pre, neg_edge_pre), dim=0).view(-1, 1)
        link_labels = torch.cat((pos_edge_pre.new_ones(pos_edge_pre.shape[0], 1),
                                 neg_edge_pre.new_zeros(neg_edge_pre.shape[0], 1)), dim=0)
        loss_test = loss_fn(link_pre, link_labels)
        acc_test, test_ratio = accuracy_node_label(link_pre, link_labels, False, False)

        auc_test, ap_test = roc_auc_score_torch(link_labels, link_pre)



    if args.supervised:
        print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test),
          "test ratio= {:.4f}".format(test_ratio),
          "f1 test = {:.4f}".format(f1_test),
          'type = {}'.format(str_name))
        with open(args.output_file+data_name,'a+') as f1:
            f1.write("Accuracy  {}, F1 {}\n".format(acc_test, f1_test))
    else:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()),
              "test ratio= {:.4f}".format(test_ratio.item()),
              "auc score = {:.4f}".format(auc_test),
              "ap score = {:.4f}".format(ap_test),
              'type = {}'.format(str_name))


def save_embeddings(embeddings):
    if args.cuda:
        embeddings = embeddings.cpu()
    embeddings = embeddings.tonumpy()
    np.save("./data/embedding_{}".format(data_name), embeddings)

if __name__ == '__main__' :
# Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


