# Implementation of K Means Clustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print('Implementation of K Means Clustering')
print('1. Make Blobs to create three Clusters; ')
print('And plot Dataset on a White Board')
X, y = make_blobs(n_samples = 150,
				  n_features = 2,
				  centers = 3,
				  cluster_std = 0.5,
				  shuffle = True,
				  random_state = 0)
plt.scatter(X[:, 0],
			X[:, 1],
			c = 'blue',
			marker = 'o',
			s = 50)
plt.grid()
plt.show()

print('\n2. Use K Means Algorithm to train the data')
print('Plot the trained data and Cluster Centers')
km = KMeans(n_clusters = 3,
			init = 'random',
			n_init = 10,
			max_iter = 300,
			tol = 1e-04,
			random_state = 0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0],
			X[y_km == 0, 1],
			s = 50,
			c = 'lightgreen',
			marker = 's',
			label = 'Cluster 0')

plt.scatter(X[y_km == 1, 0],
			X[y_km == 1, 1],
			s = 50,
			c = 'orange',
			marker = 'o',
			label = 'Cluster 1')

plt.scatter(X[y_km == 2, 0],
			X[y_km == 2, 1],
			s = 50,
			c = 'lightblue',
			marker = 'v',
			label = 'Cluster 2')

plt.scatter(km.cluster_centers_[:, 0],
			km.cluster_centers_[:, 1],
			s = 250,
			marker = '*',
			c = 'red',
			label = 'centroids')

plt.legend()
plt.grid()
plt.show()