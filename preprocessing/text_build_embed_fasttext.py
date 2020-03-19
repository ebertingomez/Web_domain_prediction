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
with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())

# Read test data
with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()

# Load the textual content of a set of webpages for each host into
# the dictionary "text".
# The encoding parameter is required since the majority of our text is french.

embedding_size = 300

treat_data = False
train_model = False

# Load texts
with open('../treated_data_nostem.json', 'r') as fp:
    text = json.load(fp)
print("Dictionnary loaded")

train_data = list()
for host in train_hosts:
    if host in text:
        train_data.append(text[host])
    else:
        train_data.append([''])

test_data = list()
for host in test_hosts:
    if host in text:
        test_data.append(text[host])
    else:
        test_data.append('')

model = gensim.models.FastText.load("model_embeddings_test_train_fastText.bin")
print("Model loaded")


# Sentence to Vect

print("Begin train data transformation")
X_train = []

for document in tqdm(train_data):
    weights = np.array([model.wv[w] if w in model.wv else np.zeros(embedding_size) 
			for w in document])
    avg_weights = np.mean(weights, axis=0)
    X_train.append(avg_weights)

print("Begin test data transformation")
X_test = []

for document in tqdm(test_data):
    weights = np.array([model.wv[w] if w in model.wv else np.zeros(embedding_size) 
			for w in document])
    avg_weights = np.mean(weights, axis=0)
    X_test.append(avg_weights)

X_train = np.array(X_train)
X_test = np.array(X_test)

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)



# Save Matrix

np.save("X_train_ff",X_train)
np.save("X_test_ff",X_test)

