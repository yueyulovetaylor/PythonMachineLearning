# Implementation of Silhouette Plot for 2 and 3 clusters to see comparison
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

print('Implementation of Silhouette plot')
print('1. Make Blobs to create three Clusters')
X, y = make_blobs(n_samples = 150,
				  n_features = 2,
				  centers = 3,
				  cluster_std = 0.5,
				  shuffle = True,
				  random_state = 0)
colors = ["Yellow", "Blue", "Green"]

print('\n2. Use 3 clusters to train the data and Plot Silhouette')
km3 = KMeans(n_clusters = 3,
			 init = "k-means++",
			 n_init = 10,
			 max_iter = 300,
			 tol = 1e-04,
			 random_state = 0)
y_km3 = km3.fit_predict(X)

cluster_labels = np.unique(y_km3)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
									 y_km3,
									 metric = 'euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
	c_silhouette_vals = silhouette_vals[y_km3 == c]
	c_silhouette_vals.sort()
	y_ax_upper += len(c_silhouette_vals)
	color = colors[i]
	plt.barh(range(y_ax_lower, y_ax_upper),
			 c_silhouette_vals,
			 height = 1.0,
			 edgecolor = 'none',
			 color = color)
	yticks.append((y_ax_upper + y_ax_lower) / 2)
	y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
			color = "red",
			linestyle = "--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.show()
print('From the Silhouette Plot, with 3 Clusters, the Coefficient is around 0.71')

print('\n3. Use 2 clusters to train the data and Plot Silhouette')
print('This is an example of Bad Cluster')
km2 = KMeans(n_clusters = 2,
			 init = "k-means++",
			 n_init = 10,
			 max_iter = 300,
			 tol = 1e-04,
			 random_state = 0)
y_km2 = km2.fit_predict(X)

print('Plot the scatter graph')
plt.scatter(X[y_km2 == 0, 0],
			X[y_km2 == 0, 1],
			s = 50,
			c = 'lightgreen',
			marker = 's',
			label = 'Cluster 1')
plt.scatter(X[y_km2 == 1, 0],
			X[y_km2 == 1, 1],
			s = 50,
			c = 'orange',
			marker = 'o',
			label = 'Cluster 2')
plt.scatter(km2.cluster_centers_[:, 0],
			km2.cluster_centers_[:, 1],
			s = 250,
			marker = '*',
			c = 'red',
			label = 'Centroids')
plt.legend()
plt.grid()
plt.show()

print('Plot the Silhouette Plot')
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
									 y_km2,
									 metric = 'euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
	c_silhouette_vals = silhouette_vals[y_km2 == c]
	c_silhouette_vals.sort()
	y_ax_upper += len(c_silhouette_vals)
	color = colors[i]
	plt.barh(range(y_ax_lower, y_ax_upper),
			 c_silhouette_vals,
			 height = 1.0,
			 edgecolor = 'none',
			 color = color)
	yticks.append((y_ax_upper + y_ax_lower) / 2)
	y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
			color = "red",
			linestyle = "--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.show()
print('From the Silhouette Plot, with 2 Clusters, the Coefficient is around 0.58')