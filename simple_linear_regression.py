#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :1].values     #remember this should be the matrix
y = dataset.iloc[:, 1].values

#spliting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

'''
#scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
'''

#fitting simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the y_test output
y_pred = regressor.predict(X_test)

#plotting the graph for Train set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Train Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#plotting the graph for Test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
