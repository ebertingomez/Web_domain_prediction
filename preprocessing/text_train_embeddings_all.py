from nltk.corpus import stopwords  # import stopwords from nltk corpus
from nltk.stem import PorterStemmer  # import the English stemming library
# import the French stemming library
from nltk.stem.snowball import FrenchStemmer
import numpy as np
from tqdm import tqdm
import os
import csv
from sklearn.linear_model import LogisticRegression
import codecs
from os import path
import gzip
import gensim
import json
import re
import nltk  # import the natural language toolkit library
nltk.download('punkt')
nltk.download('stopwords')


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

embedding_size = 300

first_train = True

with open('../treated_data.json', 'r') as fp:
    text = json.load(fp)
print("Dictionnary loaded")

filenames = os.listdir('../text/text')

set_hosts = set(filenames + train_hosts + test_hosts)
all_hosts = list(set_hosts)


all_data = list()
for host in all_hosts:
    if host in text:
        all_data.append(text[host])
    else:
        all_data.append([''])


model = gensim.models.Word2Vec(size=embedding_size, window=8, min_count=5,
                               workers=-1, sg=1, sorted_vocab=1, hs=1, seed=42,
                               batch_words=100)


if first_train:
    print("Start Training")
    model.build_vocab(all_data)
    print("Finish vocab building")
else:
    model = gensim.models.Word2Vec.load("../model_embeddings_all.bin")
    print("Model loaded")

model.train(all_data, total_examples=model.corpus_count, epochs=200,
                queue_factor=4, report_delay=3)
print("Finish Training")
model.save("../model_embeddings_all.bin")
