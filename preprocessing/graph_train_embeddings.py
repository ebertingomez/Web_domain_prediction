import csv
import numpy as np
import networkx as nx
from random import randint
import gensim

def random_walk(G, node, walk_length):
    walk = [node]
    for i in range(walk_length-1):
        neigh = list(G.neighbors(node))
        if len(neigh) > 0:
            i = randint(0, len(neigh)-1)
            node = neigh[i]
        else:
            if len(walk)==1:
                node = walk[-1]
            else:
                node = walk[-2]
        walk.append(node)
    
    walk = [str(node) for node in walk]
    return walk


def generate_walks(G, num_walks, walk_length):
    walks = [(random_walk(G, node, walk_length)) for node in G.nodes() for j in range(num_walks)]
    return walks


# Create a directed, weighted graph
G = nx.read_weighted_edgelist('../edgelist.txt', create_using=nx.DiGraph())

print(G.number_of_nodes())
print(G.number_of_edges())

first_train = False

embedding_size = 300
num_walks = 25
walk_length = 25

print("Start generating walks")
walks = generate_walks(G, num_walks, walk_length)
print("Finish generating walks")

model = gensim.models.Word2Vec(size=embedding_size, window=8, min_count=0,
                               workers=-1, sg=1, sorted_vocab=1, hs=1, seed=42,
                               batch_words=100)

if first_train:
    print("Beginnning word2vec")
    model.build_vocab(walks)
    print("Finished Vocabulary")
else:
    model = gensim.models.Word2Vec.load("../graph_embeddings_test_train.bin")
    print("Model loaded")

model.train(walks, total_examples=model.corpus_count, epochs=300,
                queue_factor=4, report_delay=3)
print("Finished Training")
model.save("../graph_embeddings_test_train.bin")
