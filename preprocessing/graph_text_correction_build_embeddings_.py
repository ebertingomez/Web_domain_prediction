import numpy as np
from tqdm import tqdm
import os
import csv
import codecs
from os import path
import gzip
import gensim
import json
import re
import networkx as nx

embedding_size = 300

G = nx.read_weighted_edgelist('../edgelist.txt', create_using=nx.Graph())

# Read training data
with open("../train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())

# Read test data
with open("../test.csv", 'r') as f:
    test_hosts = f.read().splitlines()




# Load texts
with open('../treated_data.json', 'r') as fp:
    text = json.load(fp)
print("Dictionnary loaded")


train_data = list()
wo_text_train = []
for i,host in enumerate(train_hosts):
    if host in text:
        train_data.append(text[host])
    else:
        train_data.append([''])
        wo_text_train.append(i)

test_data = list()
wo_text_test = []
for i,host in enumerate(test_hosts):
    if host in text:
        test_data.append(text[host])
    else:
        test_data.append('')
        wo_text_test.append(i)


model = gensim.models.Word2Vec.load("../model_embeddings_all.bin")
print("Model loaded")

# Sentence to Vect

filenames = os.listdir('../text/text')

set_hosts = set(filenames + train_hosts + test_hosts)
all_hosts = list(set_hosts)

# Sentence to Vect

print("Begin train data transformation")
X_all = []

all_data = list()
for host in all_hosts:
    if host in text:
        all_data.append(text[host])
    else:
        all_data.append([''])

for document in tqdm(all_data):
    weights = np.array([model.wv[w] if w in model.wv else np.zeros(embedding_size) 
			for w in document])
    avg_weights = np.mean(weights, axis=0)
    X_all.append(avg_weights)

X_all = np.array(X_all)

print("Train matrix dimensionality: ", X_all.shape)

d_all = dict(zip(all_hosts, X_all))

for i in wo_text_train:
    print(G.neighbors(train_hosts[i]))
    neigh = [d_all[n] for n in G.neighbors(train_hosts[i])]
    d_all[train_hosts[i]] = np.mean(neigh, axis=0)

for i in wo_text_test:
    print(G.neighbors(test_hosts[i]))
    neigh = [d_all[n] for n in G.neighbors(test_hosts[i])]
    d_all[test_hosts[i]] = np.mean(neigh, axis=0)


X_train = np.array([d_all[k] for k in train_hosts])
X_test = np.array([d_all[k] for k in test_hosts])

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)



# Save Matrix

np.save("../X_train_text_graph",X_train)
np.save("../X_test_text_graph",X_test)

