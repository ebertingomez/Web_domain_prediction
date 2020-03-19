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


X_train = np.load("../../X_train_text_graph.npy")
X_test = np.load("../../X_test_text_graph.npy")


# Use logistic regression to classify the webpages of the test set
clf = SVC(tol=1e-7, max_iter=2000, probability=True, kernel='rbf',
		decision_function_shape='ovo')
grid={"C":np.linspace(2.3,2.4,11)}
logreg_cv=GridSearchCV(clf, grid, cv=8, scoring="neg_log_loss", verbose=2,
            n_jobs=-1)
logreg_cv.fit(X_train,y_train)

params = logreg_cv.best_params_

print("tuned hyperparameters :(best parameters) ",params)
print("log_loss :",-logreg_cv.best_score_)


clf = SVC(tol=1e-10, max_iter=5000, probability=True,
		C=params["C"],
		kernel= 'rbf',
		decision_function_shape='ovo')

clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)

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
