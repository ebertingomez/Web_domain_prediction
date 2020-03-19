import numpy as np
from tqdm import tqdm
import os
import csv
import xgboost as xgb
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
X_test = np.load("../../X_test_text.npy")

# Use logistic regression to classify the webpages of the test set
clf = xgb.XGBClassifier(objective='multi:softprob', n_jobs=-1,
			num_parallel_tree=8, max_depth=5,
gamma=0.09000000000000001, learning_rate=0.045919801283686855, min_child_weight=5)

grid={"subsample":np.linspace(0.67,0.78,6)}
logreg_cv=GridSearchCV(clf, grid, cv=4, scoring="neg_log_loss", verbose=2, n_jobs=-1)
logreg_cv.fit(X_train,y_train)

params = logreg_cv.best_params_

print("tuned hyperparameters :(best parameters) ",params)
print("log_loss :",-logreg_cv.best_score_)


clf = xgb.XGBClassifier(objective='multi:softprob',
				learning_rate=0.045919801283686855, 
				max_depth=5, 								num_parallel_tree=8,
n_jobs=-1,
gamma=0.09000000000000001,
subsample=params['subsample'],
min_child_weight=5)

clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('text_embeddings_submit.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
