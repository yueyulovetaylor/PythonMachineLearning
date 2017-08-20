# Go through the whole process of the Principal Component Analysis
import sys
sys.path.append('../Utilities/')
import readData as RD
import numpy as np
import matplotlib.pyplot as plt

DataMap = RD.readDataFromWine()
X_train_std = DataMap['X_train_std'][0]
X_test_std = DataMap['X_test_std']
y_train = DataMap['y_train']

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

# Plot the variance explained ratio of eigenvalues
tot = sum(eigen_vals)
var_exp = [ (i / tot) for i in
		    sorted(eigen_vals, reverse = True) ]
cum_var_exp = np.cumsum(var_exp)
print('\nCumulative Variance \n%s' % cum_var_exp)

plt.bar(range(1, 14), var_exp, alpha = 0.5, align = 'center',
		label = 'individual explained variance')
plt.step(range(1, 14), cum_var_exp, where = 'mid',
		 label = 'cumulative explained variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()

# Order the eigen_pairs based eigen_values
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
			   for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse = True)
print('\n eigen pairs \n%s' % eigen_pairs)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
			   eigen_pairs[1][1][:, np.newaxis]))
print('\nTransformation Matrix W:\n%s' % w)

# Construct the PCA subspace 
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
	plt.scatter(X_train_pca[y_train == l, 0],
				X_train_pca[y_train == l, 1],
				c = c, label = l, marker = m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.show()