# Artifical Neural Network

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv');
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# handling the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Building ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier = Sequential()
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# predicting the results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# predicitin a new single observation
new_observation = classifier.predict(sc_X.transform(np.array([[0, 0, 600, 1, 40, 3, 6000, 2, 1, 1, 5000]])))
new_observation = (new_observation > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# k fold cross validation
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
     classifier = Sequential()
     classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
     classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
     return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
accuracies.mean()
accuracies.std()

# Grid Search
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
     classifier = Sequential()
     classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
     return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25, 32],
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_ 