from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def printDecisionRegion(X, y, classifier, resolution = 0.02):
	# Utility function to plot decision region on a two dimension coordination
	# Set up marker and color set
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')

	cmap = ListedColormap(colors[:len(np.unique(y))])
		# Get a list of unique y's and parse into color array to get 
		# unique colors constructing Listed Colormap

	# Plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	# Use python to set up the coordination and do calculation via every 
	# pixel points to get the classification data
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
		np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)

	plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# Plot class samples
	for idx, cl in enumerate(np.unique(y)):
		# cl is the what the cluster is
		plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
			        alpha = 0.8, c = cmap(idx), marker = markers[idx], 
			        label = cl)

def printDecisionRegionWithTestIdx(X, y, classifier, test_idx = None, resolution = 0.02):
	# Utility function to plot decision region on a two dimension coordination
	# Set up marker and color set
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')

	cmap = ListedColormap(colors[:len(np.unique(y))])
		# Get a list of unique y's and parse into color array to get 
		# unique colors constructing Listed Colormap

	# Plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	# Use python to set up the coordination and do calculation via every 
	# pixel points to get the classification data
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
		np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)

	plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# Plot class samples
	for idx, cl in enumerate(np.unique(y)):
		# cl is the what the cluster is
		plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
			        alpha = 0.8, c = cmap(idx), marker = markers[idx], 
			        label = cl)

	# Highlight all test samples
	if test_idx:
		print('plot test')
		X_test, y_test = X[test_idx, :], y[test_idx]
		print(X_test)
		print(y_test)
		plt.scatter(X_test[:, 0], X_test[:, 1], edgecolors = 'black', 
			        facecolors = 'none', alpha = 1.0, linewidths = 1, 
			        marker = 'o', s = 80, label = 'test set')
