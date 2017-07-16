# Implementation of L2 Regularization in Logistic Regression
import numpy as np
import sys
sys.path.append('../Utilities/')
import sklearnReadData as skRead
from sklearn.linear_model import LogisticRegression
import math
import matplotlib.pyplot as plt

print('Implementation of L2 Regularization in Logistic Regression')

# Use utility API to read data out of sklearn library
dataSet = skRead.sklearnReadIris()
X_train_std = dataSet['X_train_std']
y_train = dataSet['y_train']
X_test_std = dataSet['X_test_std']
y_test = dataSet['y_test']

# Define weights and parameters (C)
weights = []
params = []
for c in np.arange(-5, 5):
	param = math.pow(10, c)
	lr = LogisticRegression(C = param, random_state = 0)
	lr.fit(X_train_std, y_train)
	weights.append(lr.coef_[1])    # lr.coef[1] includes weights of the features
	params.append(param)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label = 'petal length')
plt.plot(params, weights[:, 1], label = 'petal width', linestyle = '--')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()
