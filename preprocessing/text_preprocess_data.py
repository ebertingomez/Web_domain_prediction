from nltk.corpus import stopwords  # import stopwords from nltk corpus
from nltk.stem import PorterStemmer  # import the English stemming library
from nltk.stem.snowball import FrenchStemmer # import the French stemming library
import numpy as np
from tqdm import tqdm
import os
import csv
import codecs
from os import path
import gzip
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


text = dict()
filenames = os.listdir('../text/text')
for filename in tqdm(filenames):
    with codecs.open(path.join('../text/text/', filename),
                        encoding='latin-1') as f:
        raw = f.read().replace("\n", " ").lower()
        no_commas = re.sub(r"[^a-z0-9]+", ' ', raw)
        # remove the excess whitespace from the raw text
        no_spaces = re.sub(r'\s+|\t+', ' ', no_commas)
        # generate a list of tokens from the raw text
        tokens = nltk.word_tokenize(no_spaces)
        # create a nltk text from those tokens
        tokens = nltk.Text(tokens, 'latin-1')
        filtered_fr = [w for w in tokens if not w in stopwords_fr
                        and w.isalpha() and len(w) > 1]
        filtered_en = [w for w in filtered_fr if not w in stopwords_en
                        and w.isalpha() and len(w) > 1]
        stem_fr = [stemmer_fr.stem(w) for w in filtered_en]
        stem_en = [stemmer_en.stem(w) for w in stem_fr]
        text[filename] = stem_en if stem_en else ['']

with open('../treated_data.json', 'w') as fp:
    json.dump(text, fp)