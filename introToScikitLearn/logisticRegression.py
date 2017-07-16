# Implementation of Logistic Regression
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../Utilities/')
import sklearnReadData as skRead
import printDecisionRegion as printDR
from sklearn.linear_model import LogisticRegression

# First, we use numpy and matplotlib to plot the shape of a sigmoid function
print('Print Signoid Function where X ranges from -7 to 7')
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
	# Return evenly spaced values within a given interval as an array
plt.plot(z, phi_z)

# Plot line x = 0; and two limit line y = 1 and y = -1
plt.axvline(0.0, color = 'k')
plt.axhline(0.0, ls = 'dotted', color = 'k')
plt.axhline(0.5, ls = 'dotted', color = 'k')
plt.axhline(1.0, ls = 'dotted', color = 'k')

plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()

# Use utility API to read data out of sklearn library
dataSet = skRead.sklearnReadIris()
X_train_std = dataSet['X_train_std']
y_train = dataSet['y_train']
X_test_std = dataSet['X_test_std']
y_test = dataSet['y_test']

# Logistic Regression train the data and plot the decision region
lr = LogisticRegression(C = 1000.0, random_state = 0)
lr.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
printDR.printDecisionRegionWithTestIdx(X = X_combined_std, y = y_combined, 
	                                   classifier = lr, test_idx = range(105, 150))
plt.xlabel('petal length [Standarized]')
plt.ylabel('petal width [Standarized]')
plt.legend(loc = 'upper left')
plt.show()

probArray = lr.predict_proba(X_test_std[0, :])
print('Test probabilities array (The third member is the probability of accuracy)', 
	   probArray)
