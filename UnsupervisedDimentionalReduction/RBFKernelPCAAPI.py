# Implementation of RBF Kernel PCA API
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
	print('Implementation of the RBF Kernel PCA API')

	# Calculate pairwise squared distance and transform the matrix into 
	# NxN squared matrix
	sq_dist = pdist(X, 'sqeuclidean')
	mat_sq_dists = squareform(sq_dist)
	print('Size of mat_sq_dists are %s x %s' % 
		  (mat_sq_dists.shape[0], mat_sq_dists.shape[1]))

	# Compute the symmetric kernel matrix and center the kernel matrix
	K = exp(-gamma * mat_sq_dists)
	N = K.shape[0]
	one_n = np.ones((N, N)) / N
	K = K -one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	# Calculate the eigenpairs of K and get the X_pc matrix
	eigvals, eigvecs = eigh(K)
	X_pc = np.column_stack(eigvecs[:, -i]
		                   for i in range(1, n_components + 1))
	return X_pc

def rbf_kernel_pca_return_K_alpha(X, gamma, n_components):
	print('Implementation of the RBF Kernel PCA API II')
	print('This API returns the eigenvector Alpha and eigenvalue Lamda')

	# Calculate pairwise squared distance and transform the matrix into 
	# NxN squared matrix
	sq_dist = pdist(X, 'sqeuclidean')
	mat_sq_dists = squareform(sq_dist)
	print('Size of mat_sq_dists are %s x %s' % 
		  (mat_sq_dists.shape[0], mat_sq_dists.shape[1]))

	# Compute the symmetric kernel matrix and center the kernel matrix
	K = exp(-gamma * mat_sq_dists)
	N = K.shape[0]
	one_n = np.ones((N, N)) / N
	K = K -one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	# Calculate the eigenpairs of K and get the X_pc matrix
	eigvals, eigvecs = eigh(K)

	# Eigen vector
	print('Collect eigenvectors Alphas')
	alphas = np.column_stack(eigvecs[:, -i]
		                    for i in range(1, n_components + 1))
	print('alpha is \n%s' % alphas)

	# Eigen value
	print('Collect eigenvalues Lambdas')
	lambdas = [eigvals[-i] 
	           for i in range(1, n_components + 1)]
	print('lambdas is \n%s' % lambdas[0])

	return alphas, lambdas
