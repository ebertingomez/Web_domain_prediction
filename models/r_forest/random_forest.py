import numpy as np
from tqdm import tqdm
import os
import csv
from sklearn.ensemble import RandomForestClassifier
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
clf = RandomForestClassifier(max_depth=50, max_features=220, n_jobs=-1)
grid={"n_estimators":[365, 375, 385], "min_impurity_decrease":np.logspace(-4,-2,3), 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],}
logreg_cv=GridSearchCV(clf, grid, cv=8, scoring="neg_log_loss", verbose=2)
logreg_cv.fit(X_train,y_train)

params = logreg_cv.best_params_

print("tuned hyperparameters :(best parameters) ",params)
print("log_loss :",-logreg_cv.best_score_)


clf = RandomForestClassifier(max_depth=50,
				n_estimators=params["n_estimators"], 
				min_impurity_decrease =params["min_impurity_decrease"], 
				max_features=220,
n_jobs=1,
min_samples_split=params["min_samples_split"],
min_samples_leaf=params['min_samples_leaf'])

clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('text_rf_submit.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
