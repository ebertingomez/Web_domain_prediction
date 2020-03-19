import numpy as np
from tqdm import tqdm
import os
import csv
from sklearn.linear_model import LogisticRegression
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

# Use logistic regression to classify the webpages of the test set
clf = LogisticRegression(multi_class='auto', max_iter=2000, 
			 tol=1e-7)
grid={"C":np.logspace(-3,3,7)}
logreg_cv=GridSearchCV(clf, grid, cv=8, scoring="neg_log_loss", verbose=2,
            n_jobs=-1)
logreg_cv.fit(X_train,y_train)

params = logreg_cv.best_params_

print("tuned hyperparameters :(best parameters) ",params)
print("log_loss :",-logreg_cv.best_score_)


clf = LogisticRegression(multi_class='auto', max_iter=3000,
			 tol=1e-7, C=params["C"])
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('text_logreg_submit.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
