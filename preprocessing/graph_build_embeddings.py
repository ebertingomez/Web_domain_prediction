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

# Load the textual content of a set of webpages for each host into
# the dictionary "text".
# The encoding parameter is required since the majority of our text is french.


model = gensim.models.Word2Vec.load("../graph_embeddings_test_train.bin")
print("Model loaded")


# Sentence to Vect

print("Begin train data transformation")
X_train = []

for node in tqdm(train_hosts):
    X_train.append(model.wv[node])

print("Begin test data transformation")
X_test = []

for node in tqdm(test_hosts):
    X_test.append(model.wv[node])

X_train = np.array(X_train)
X_test = np.array(X_test)

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)



# Save Matrix

np.save("../X_train_graph",X_train)
np.save("../X_test_graph",X_test)

