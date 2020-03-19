import csv
import torch
import networkx as nx
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from models import GNN
import time
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
from sklearn.manifold import TSNE
from scipy import sparse
from torch_geometric import data, io
from torch_geometric.utils import from_networkx


# Hyperparameters
epochs = 150
n_hidden_1 = 20
n_hidden_2 = 20
learning_rate = 0.01
dropout_rate = 0.2
n_layers = 2

# Read training data
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
class_labels = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    class_labels.append(label.lower())
train_hosts = list(np.unique(train_hosts))
n_train_host = len(train_hosts)
n_class = 8

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return classes, labels_onehot

classes, labels = encode_onehot(class_labels)

# Read test data
with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
n_test_host = len(np.unique(train_hosts))

n = n_train_host + n_test_host

hosts_list = train_hosts + test_hosts

print('Creating networkx graph from hosts')
# Create a directed, weighted graph with the annotated nodes
G = nx.read_weighted_edgelist('edgelist.txt')

node_l = G.nodes()
n_nodes = len(node_l)
node_l = [x for x in node_l if x not in hosts_list]
nodelist = hosts_list + node_l

''' Uncomment to load the precomputed embeding as features
# loadading embeding
print('Loading embeding')
X = np.load('X_all.npy')
node_emb = X[:,0].astype('int')
node_emb = node_emb.astype('str')
X = X[:,1:]

'''

## Computing features
print('Computing pagerank')
pagerank = nx.pagerank(G)
print('Computing triangles')
triangles = nx.triangles(G)
print('Computing degree centrality')
deg_centrality = nx.degree_centrality(G)
print('Computing Core number')
core_number = nx.core_number(G)
print('Computing color number')
color_number = nx.algorithms.coloring.greedy_color(G)

# Computing feature matrix
features = []
features_dict = dict()
for i in nodelist:
    features.append([pagerank[i], triangles[i], deg_centrality[i], core_number[i], color_number[i]])
    features_dict[i] = np.array([pagerank[i], triangles[i], deg_centrality[i], core_number[i], color_number[i]])

features = np.array(features)
nx.set_node_attributes(G, name ='x',values = features_dict)


print('Normalizing Laplacian')
## Peut-être que la normalisation n'est pas bonne
adj = nx.normalized_laplacian_matrix(G, nodelist=nodelist).tocoo()
values = adj.data
indices = np.vstack((adj.row, adj.col))

# Yields indices to split data into training, validation and test sets
features = torch.FloatTensor(features)
idx = np.random.permutation(n_train_host)
idx_train = idx[:int(0.8*n_train_host)]
idx_val = idx[int(0.8*n_train_host):]
idx_test = np.arange(n_train_host, n)

# Transform the numpy matrices/vectors to torch tensors
y = torch.LongTensor(np.argmax(labels, axis=1))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
# .todense(?)
adj = torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

print('Initializing GNN')
# Creates the model and specifies the optimizer
model = GNN(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(epoch):

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], y[idx_train])
    acc_train = accuracy(output[idx_train], y[idx_train])
    loss_train.backward()
    optimizer.step()
    
    model.eval()
    out = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], y[idx_val])
    acc_val = accuracy(output[idx_val], y[idx_val])

    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    return torch.exp(output[idx_test])


print('Starting training')
# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print()

# Testing
y_pred = test()

print('Saving test data')
# Write predictions to a file
with open('gnn_pred.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = list(classes)
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)

print('Work done')
