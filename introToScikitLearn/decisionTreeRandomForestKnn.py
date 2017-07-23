# Implementation of Decision Tree, Random Forest and K nearest neighbors
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append('../Utilities/')
import printDecisionRegion as printDR
import sklearnReadData as skRead
import matplotlib.pyplot as plt

print('Training a decision tree classification via scikit-learn')

# Use utility API to read data out of sklearn library
dataSet = skRead.sklearnReadIris()
X_train = dataSet['X_train']
X_train_std = dataSet['X_train_std']
y_train = dataSet['y_train']
X_test = dataSet['X_test']
X_test_std = dataSet['X_test_std']
y_test = dataSet['y_test']
X_combined = np.vstack((X_train, X_test))
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Construct the Decision Tree and Plot the decision region
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
tree.fit(X_train, y_train)

printDR.printDecisionRegionWithTestIdx(X = X_combined, y = y_combined, 
	                                   classifier = tree, test_idx = range(105, 150))
plt.xlabel('petal length [Standarized]')
plt.ylabel('petal width [Standarized]')
plt.legend(loc = 'upper left')
plt.title('Decision Tree Classify Iris')
plt.show()

print('Training a decision tree classification via scikit-learn')
forest = RandomForestClassifier(criterion = 'entropy', 
								n_estimators = 10, 
								random_state = 1,
								n_jobs = 2)
forest.fit(X_train, y_train)
printDR.printDecisionRegionWithTestIdx(X = X_combined, y = y_combined, 
	                                   classifier = forest, test_idx = range(105, 150))
plt.xlabel('petal length [Standarized]')
plt.ylabel('petal width [Standarized]')
plt.legend(loc = 'upper left')
plt.title('Random Forest Classify Iris')
plt.show()

print('Training a K nearest Neighbors classification via scikit-learn')
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
knn.fit(X_train_std, y_train)
printDR.printDecisionRegionWithTestIdx(X = X_combined_std, y = y_combined, 
	                                   classifier = knn, test_idx = range(105, 150))
plt.xlabel('petal length [Standarized]')
plt.ylabel('petal width [Standarized]')
plt.legend(loc = 'upper left')
plt.title('Knn Classify Iris')
plt.show()