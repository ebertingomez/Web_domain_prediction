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

stemmer_fr = PorterStemmer()
stemmer_en = FrenchStemmer()
stopwords_fr = stopwords.words('french')
stopwords_en = stopwords.words('english')
embedding_size = 300

first_train = False

with open('../treated_data.json', 'r') as fp:
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
        test_data.append([''])

all_data = train_data + test_data


model = gensim.models.Word2Vec(size=embedding_size, window=8, min_count=5,
                               workers=6, sg=1, sorted_vocab=1, hs=1, seed=42,
                               batch_words=100)


if first_train:
    print("Start Training")
    model.build_vocab(all_data)
    print("Finish vocab building")
else:
    model = gensim.models.Word2Vec.load("../model_embeddings_test_train.bin")
    print("Model loaded")

model.train(all_data, total_examples=model.corpus_count, epochs=150,
                queue_factor=4, report_delay=3)
print("Finish Training")
model.save("../model_embeddings_test_train.bin")
