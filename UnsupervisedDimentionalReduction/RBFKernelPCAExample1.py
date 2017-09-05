# Sample 1 RBF Kernel PCA using make_moons sample
import sys
sys.path.append('./')
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import RBFKernelPCAAPI as RBFAPI

print('Kernel Trick -- Sample 1 RBF Kernel PCA using make_moons sample')
print('1. Read make_moons data and plot the scatter')
X, y = make_moons(n_samples = 100, random_state = 123)
plt.scatter(X[y == 0, 0], X[y == 0, 1], 
	        color = 'red', marker = '^', alpha = 0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], 
	        color = 'blue', marker = 'o', alpha = 0.5)
plt.show()

print('\n2. Use PCA try to separate the data')
print('Clearly, it cannot be linear separated')
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))

# Plot PC1 vs PC2
ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
	          color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
	          color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')

# Mapping only to PC1
ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
	          color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
	          color = 'blue', marker = 'o', alpha = 0.5)
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

print('\n3. Use the RBF Kernel PCA to transform the X into new space')
print('Which can be clearly linear separated')
X_kpca = RBFAPI.rbf_kernel_pca(X, gamma = 15, n_components = 2)
print('X_kernel PCA size %s x %s' % (X_kpca.shape[0], X_kpca.shape[1]))

print('Plot the scatter on RBF PC1 and PC2')
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3))

ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
	          color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
	          color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')

# Mapping only to PC1
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02,
	          color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02,
	          color = 'blue', marker = 'o', alpha = 0.5)
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.show()

print('\n4. Call API II to collect eigenvalues and eigenvectors, projecting one sample points')
alphas, lambdas = RBFAPI.rbf_kernel_pca_return_K_alpha(X, gamma = 15, n_components = 1)

# Use the 25th sample in X as the one to do project 
X_origin = X[25]
X_project = alphas[25]

# Create a throw-in API to use the distance function to project
def project_x(x_origin, X, gamma, alphas, lambdas):
	pair_dist = np.array([np.sum((x_origin - row) ** 2)
		                 for row in X])
	k = np.exp(-gamma * pair_dist)
	return k.dot(alphas / lambdas)
X_reproject = project_x(X_origin, X, gamma = 15, alphas = alphas, lambdas = lambdas)

print('Sample Example %s project to %s (Use API to calculate remap value: %s)' 
	  % (X_origin, X_project, X_reproject))

# Visualize the remapping process
plt.scatter(alphas[y == 0, 0], np.zeros((50)), color = 'red', marker = '^', alpha = 0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50)), color = 'blue', marker = 'o', alpha = 0.5)
plt.scatter(X_project, 0, color = 'black', 
	        label = 'original projection of point X[25]', 
	        marker = '^', s = 100)
plt.scatter(X_reproject, 0, color = 'green', 
	        label = 'remap point X[25]', 
	        marker = 'x', s = 500)
plt.legend(scatterpoints = 1)
plt.show()

print('\n5. Sklearn PCA Application')
from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1], 
	        color = 'red', marker = '^', alpha = 0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1], 
	        color = 'blue', marker = 'o', alpha = 0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
