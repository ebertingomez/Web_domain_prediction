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


X_train = np.load("../../X_train_text.npy")
X_test = np.load("../../X_train_text.npy")

filenames = os.listdir('../text/text')

clf = SVC(tol=1e-10, max_iter=10000, probability=True,
		C=1.3557142857142859,
        gamma=0.296,
		kernel= 'rbf',
		decision_function_shape='ovo')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Write predictions to a file
with open('text_svc_submit_ff.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
