import numpy as np
from tqdm import tqdm
import os
import csv
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from os import path


# Read training data
with open("../../train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())

# Read test data
with open("../../test.csv", 'r') as f:
    test_hosts = f.read().splitlines()


X_train_text = np.load("../../X_train_text.npy")
X_test_text = np.load("../../X_train_text.npy")


X_train_graph = np.load("../../X_train_graph.npy")
X_test_graph = np.load("../../X_test_graph.npy")

# Text

clf = SVC(tol=1e-10, max_iter=10000, probability=True,
		C=1.3557142857142859,
		kernel= 'rbf',
        gamma=0.296,
		decision_function_shape='ovo')

clf.fit(X_train_text, y_train)

y_pred_text = clf.predict_proba(X_test_text)

# Graph

clf = SVC(tol=1e-10, max_iter=10000, probability=True,
		C=5711.111111111111,
		kernel= 'rbf',
		decision_function_shape='ovo')

clf.fit(X_train_graph, y_train)

y_pred_graph = clf.predict_proba(X_test_graph)

y_pred = (y_pred_text + y_pred_graph)/2

# Write predictions to a file
with open('text_graph_submit.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
