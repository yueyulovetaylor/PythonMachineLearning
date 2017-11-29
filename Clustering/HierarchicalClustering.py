# Implementation of Hierarchical Clustering using Complete Linkage
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

print('Implementation of Hierarchical Clustering using Complete Linkage')
print('1. Initiate 5 numbers for Clustering')
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns = variables, index = labels)
print('Data Frame initiated as ')
print(df)

print('\n2. Construct the Distance Matrix')
row_dist = pd.DataFrame(squareform(
							pdist(df, metric = 'euclidean')),
						columns = labels,
						index = labels)
print('Distance Matrix shown as')
print(row_dist)

print('\n3. Complete Linkage to cluster the data points')
row_clusters = linkage(df.values, method = 'complete', metric = 'euclidean')
row_cluster_df = pd.DataFrame(row_clusters,
			 columns = ['Row Label 1',
			 			'Row Label 2',
			 			'Distance',
			 			'# of items in the Cluster'],
			  index = ['Cluster %d' %(i+1) for i in range(row_clusters.shape[0])])
print('Result of Hierarchical Cluster')
print(row_cluster_df)

print('\n4. Plot the Dendrogram')
row_dendr = dendrogram(row_clusters,
					   labels = labels)
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()

print('\n5. Attach the Heat Map')
fig = plt.figure(figsize = (8, 8), facecolor = 'white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation = 'left')

df_rowclust = df.ix[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
				  interpolation = 'nearest',
				  cmap = 'hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
	i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

print('\n6. Sklearn to train the data')
ac = AgglomerativeClustering(n_clusters = 2,
							 affinity = "euclidean",
							 linkage = "complete")
labels = ac.fit_predict(X)
print("SKLearn Prediction Result:")
print(labels)