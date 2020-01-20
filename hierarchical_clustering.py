# heirarchical clustering

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#making the dendograms for deciding the number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.show()

# fitting the hierarchy to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# visualizing the results
plt.scatter(X[y_hc == 0 ,0], X[y_hc == 0 ,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1 ,0], X[y_hc == 1 ,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2 ,0], X[y_hc == 2 ,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3 ,0], X[y_hc == 3 ,1], s = 100, c = 'pink', label = 'Cluster 4')
plt.scatter(X[y_hc == 4 ,0], X[y_hc == 4 ,1], s = 100, c = 'cyan', label = 'Cluster 5')
plt.legend()
plt.show()