import numpy as np
from tqdm import tqdm
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from os import path


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

from sklearn import preprocessing



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


X_train_text = np.load("../../X_train.npy")
X_test_text = np.load("../../X_test.npy")

X_train_graph = np.load("../../X_train_graph.npy")
X_test_graph = np.load("../../X_test_graph.npy")

X_train = np.column_stack((X_train_text,X_train_graph))
X_test = np.column_stack((X_test_text,X_test_graph))

print(X_train.shape)
print(X_test.shape)

le = preprocessing.LabelEncoder()
y_number = le.fit_transform(y_train)
categorical_labels = to_categorical(y_number, num_classes=None)

model = Sequential()
model.add(Dense(600, input_dim=600))
model.add(Activation('relu'))

#model.add(Dropout(0.5))
model.add(Dense( len(le.classes_) ))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])


model.fit(X_train, categorical_labels,
          epochs=100,
          batch_size=20,
          workers=-1,
          validation_split=0.20
          )

model.save_weights('../../text_graph_fc_model.h5')

y_pred = model.predict_proba(X_test)

# Write predictions to a file
with open('text_mlp_submit.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = le.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)