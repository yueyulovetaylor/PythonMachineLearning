# Training a svm classification via scikit-learn
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../Utilities/')
import printDecisionRegion as printDR
import sklearnReadData as skRead
import matplotlib.pyplot as plt

print('Training a svm classification via scikit-learn')

# Use utility API to read data out of sklearn library
dataSet = skRead.sklearnReadIris()
X_train_std = dataSet['X_train_std']
y_train = dataSet['y_train']
X_test_std = dataSet['X_test_std']
y_test = dataSet['y_test']

svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
printDR.printDecisionRegionWithTestIdx(X = X_combined_std, y = y_combined, 
	                                   classifier = svm, test_idx = range(105, 150))
plt.xlabel('petal length [Standarized]')
plt.ylabel('petal width [Standarized]')
plt.legend(loc = 'upper left')
plt.show()