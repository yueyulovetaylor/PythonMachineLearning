# Sample 2 RBF Kernel PCA using make_circles sample
import sys
sys.path.append('./')
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import RBFKernelPCAAPI as RBFAPI

print('Kernel Trick -- Sample 1 RBF Kernel PCA using make_circles sample')
X, y = make_circles(n_samples = 1000, random_state = 123, noise = 0.1, factor = 0.2)
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
ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,
	          color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,
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
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02,
	          color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02,
	          color = 'blue', marker = 'o', alpha = 0.5)
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.show()