# here we are doing k means clustering algo

# importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# preprocessing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# finding the right amount of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_);
plt.plot(range(1, 11), wcss)
plt.show()

# making the cluster with clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_pred = kmeans.fit_predict(X)

# visiualizing the result
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c = 'red', s = 100, label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c = 'green', s = 100, label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], c = 'blue', s = 100, label = 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], c = 'pink', s = 100, label = 'Cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], c = 'cyan', s = 100, label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', s = 300, label = 'cenetroids')
plt.legend()
plt.show()

