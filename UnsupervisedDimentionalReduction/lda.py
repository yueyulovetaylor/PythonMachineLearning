# Implementation of the Linear Discriminant Analysis
import sys
sys.path.append('../Utilities/')
import readData as RD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
import printDecisionRegion as printDR

print("Implementation of the Linear Discriminant Analysis")
print("1. Call ReadData API to get the Data")
DataMap = RD.readDataFromWine()
X_train_std = DataMap['X_train_std'][0]
X_test_std = DataMap['X_test_std']
y_train = DataMap['y_train']
y_test = DataMap['y_test']

np.set_printoptions(precision = 4)
mean_vecs = []

# Calculate the mean vector of each feature
print("\n2. Calculate the mean vector of each feature")
for label in range(1, 4):
	mean_vecs.append(np.mean(
		X_train_std[y_train == label], axis = 0))
	print('MV %s: %s' %(label, mean_vecs[label - 1]) )

# Compute the within scatter matrix
print("\n3. Compute the within scatter matrix")
# Method 1: Directly compute the unscaled Scatter Matrix
print("Method 1: Directly compute the unscaled Scatter Matrix")
d = 13    # Number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
	class_scatter = np.zeros((d, d))
	for row in X_train_std[y_train == label]:
		row, mv = row.reshape(d, 1), mv.reshape(d, 1)
		class_scatter += (row - mv).dot((row - mv).T)
	S_W += class_scatter

print("Within-class scatter matrix: %s*%s" % (S_W.shape[0], S_W.shape[1]))

# Print the size of each class to check whether they are normally distributed
print("Class distribution %s" % np.bincount(y_train)[1:])

# Method 2: Use the Covariance Matrix to get the Scaled Scatter Matrix
print("\nMethod 2: If we scaled Scatter Matrix by dividing class sample number Ni, " +
	  "we find the Scaled Scatter Matrix is actually the Covariance Matrix")
S_W_scaled = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
	class_scatter_scaled = np.cov(X_train_std[y_train == label].T)
	S_W_scaled += class_scatter_scaled

print("Scaled within-class scatter matrix: %s*%s" % (S_W.shape[0], S_W.shape[1]))

# Compute the Between Class Scatter Matrix
print("\n3. Compute the Between Class Scatter Matrix")
S_B = np.zeros((d, d))
mean_overall = np.mean(X_train_std, axis = 0)
for i, mean_vecs in enumerate(mean_vecs):
	n = X_train_std[y_train == i + 1, :].shape[0]
	mean_vecs = mean_vecs.reshape(d, 1)
	mean_overall = mean_overall.reshape(d, 1)
	S_B += n * (mean_vecs - mean_overall).dot((mean_vecs - mean_overall).T)

print("Between-class scatter matrix: %s*%s" % (S_B.shape[0], S_B.shape[1]))

# Calculate the eigenvalues of (S_W^-1 (dot) S_B)
print("\n4. Calculate the eigenvalues of (S_W^-1 (dot) S_B) to get the new feature subspace")
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W_scaled).dot(S_B))
eigen_paris = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
			   for i in range(len(eigen_vals))]
# Sort the eigen pairs
eigen_paris = sorted(eigen_paris, key = lambda k: k[0], reverse = True)
print('Eigenvalues in decreasing order:')
for eigen_val in eigen_paris:
	print(eigen_val[0])

print('\nNumber of Linear Discriminant is at most(c - 1), in this case 2')
print('Plot the Linear Discriminant by decreasing orders')
total = sum(eigen_vals.real)
discr = [(i / total) for i in sorted(eigen_vals.real, reverse = True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha = 0.5, align = 'center', 
		label = 'Individual "Discriminants"')
plt.step(range(1, 14), cum_discr, where = 'mid',
		label = 'Cumulative "Discriminants"')
plt.ylabel('"Discriminality" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc = 'best')
plt.show()

print("\nCreate transformation matrix by using the two most discriminative eigenvector columns")
W = np.hstack((eigen_paris[0][1][:, np.newaxis].real,
			   eigen_paris[1][1][:, np.newaxis].real))
print('Transformation Matrix W\n', W)

# Project samples onto the new Sample space
print("\n5. Project samples onto the new Sample space")
X_train_lda = X_train_std.dot(W)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
	plt.scatter(X_train_lda[y_train == l, 0] * (-1),
				X_train_lda[y_train == l, 1] * (-1),
				c = c, label = l, marker = m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower right')
plt.show()

# LDA via sklearn
print("\n6. LDA via sklearn")
lda = LDA(n_components = 2)
X_train_sk_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression()
lr = lr.fit(X_train_sk_lda, y_train)

# Print the Training Dataset
print("Print the Decision Region of Training Dataset")
printDR.printDecisionRegion(X_train_sk_lda, y_train, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower right')
plt.show()

# Print the Test Dataset
X_test_sk_lda = lda.transform(X_test_std)
print("Print the Decision Region of Testing Dataset")
printDR.printDecisionRegion(X_test_sk_lda, y_test, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower right')
plt.show()