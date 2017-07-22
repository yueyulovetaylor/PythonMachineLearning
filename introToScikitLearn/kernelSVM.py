# Implementtion of Kernel SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import sys
sys.path.append('../Utilities/')
import printDecisionRegion as printDR
import sklearnReadData as skRead

# Plot an Xor graph on the coordination
print('Print an XOR logic graph on the coordination')
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
	        c = 'b', marker = 'x', label = '1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], 
	        c = 'r', marker = 's', label = '-1')
plt.ylim(-3.0)
plt.legend()
plt.title('XOR for random data')
plt.show()

# Plot Kernel (RBF) SVM on the Xor data set
print('Plot Kernel (RBF) SVM on the Xor data set')
svm = SVC(kernel = 'rbf', random_state = 0, gamma = 0.10, C = 10.0)
svm.fit(X_xor, y_xor)
printDR.printDecisionRegionWithTestIdx(X_xor, y_xor, classifier = svm)
plt.legend(loc = 'upper left')
plt.title('Kernel (RBF) SVM on the Xor data set')
plt.show()

# Plot Kernel on Iris Data 
dataSet = skRead.sklearnReadIris()
X_train_std = dataSet['X_train_std']
y_train = dataSet['y_train']
X_test_std = dataSet['X_test_std']
y_test = dataSet['y_test']

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

print('Plot Kernel (RBF) SVM on the Iris data set')
svm = SVC(kernel = 'rbf', random_state = 0, gamma = 0.2, C = 1.0)
svm.fit(X_train_std, y_train)
printDR.printDecisionRegionWithTestIdx(X = X_combined_std, y = y_combined, 
	                                   classifier = svm, test_idx = range(105, 150))
plt.legend(loc = 'upper left')
plt.title('Kernel (RBF) SVM on the Iris data set')
plt.show()