#ploynomial regression 

#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#making a linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#making a polynomial linear regression model
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree = 4)
X_poly = pol_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

y_pred_1 = lin_reg.predict(X)
y_pred_2 = lin_reg_2.predict(X_poly)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred_1, color = 'blue')
plt.plot(X_grid, lin_reg_2.predict(pol_reg.fit_transform(X_grid)), color = 'green')
plt.show()