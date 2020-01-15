# random forest regression

#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv');
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#making a regressor for random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000)
regressor.fit(X, y)

#predicting a salary of level 6.5
y_pred = regressor.predict([[6.5]]);

#plotting the graph in high resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue');
