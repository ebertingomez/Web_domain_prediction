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

# Load the textual content of a set of webpages for each host into
# the dictionary "text".
# The encoding parameter is required since the majority of our text is french.

embedding_size = 300

# Load texts
with open('../treated_data.json', 'r') as fp:
    text = json.load(fp)
print("Dictionnary loaded")

model = gensim.models.Word2Vec.load("../model_embeddings_all.bin")
print("Model loaded")

### Generation of all hosts in the dataset
# Read training data
with open("../train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)

# Read test data
with open("../test.csv", 'r') as f:
    test_hosts = f.read().splitlines()

filenames = os.listdir('../text/text')

set_hosts = set(filenames + train_hosts + test_hosts)
print(len(filenames))
print(len(set_hosts))
all_hosts = list(set_hosts)
print(len(all_hosts))

# Sentence to Vect

print("Begin train data transformation")
X_all = []

train_data = list()
for host in all_hosts:
    if host in text:
        train_data.append(text[host])
    else:
        train_data.append([''])

for document in tqdm(train_data):
    weights = np.array([model.wv[w] if w in model.wv else np.zeros(embedding_size) 
			for w in document])
    avg_weights = np.mean(weights, axis=0)
    X_all.append(avg_weights)

X_all = np.array(X_all)

print("Train matrix dimensionality: ", X_all.shape)

all_hosts = np.asarray(all_hosts, dtype=np.int)

X_all = np.column_stack((all_hosts, X_all))

# Save Matrix

np.save("../X_all",X_all)

