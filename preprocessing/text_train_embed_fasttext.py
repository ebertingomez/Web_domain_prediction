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

stemmer_fr = PorterStemmer()
stemmer_en = FrenchStemmer()
stopwords_fr = stopwords.words('french')
stopwords_en = stopwords.words('english')
embedding_size = 300

treat_data = False
train_model = True
first_train = True

if treat_data:
    text = dict()
    filenames = os.listdir('text/text')
    for filename in tqdm(filenames):
        with codecs.open(path.join('text/text/', filename),
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

    with open('treated_data.json', 'w') as fp:
        json.dump(text, fp)
else:
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
        test_data.append([''])

all_data = train_data + test_data


model = FastText(size=embedding_size, window=8, min_count=3,
                               workers=-1, sg=1, sorted_vocab=1, hs=1, seed=42,
                               batch_words=100,word_ngrams=1)


if train_model:
    print("Start Training")
    if first_train:
        model.build_vocab(all_data)
        print("Finish vocab building")
    else:
        model = gensim.models.FastText.load("model_embeddings_test_train_fastText.bin")
        
    model.train(all_data, total_examples=model.corpus_count, epochs=500,
                queue_factor=4, report_delay=3)
    print("Finish Training")
    model.save("model_embeddings_test_train_fastText.bin")
else:
    model = gensim.models.FastText.load("model_embeddings_test_train_fastText.bin")
    print("Model loaded")
