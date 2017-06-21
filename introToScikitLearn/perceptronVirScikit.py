# Training a perceptron via scikit-learn
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

import sys
sys.path.append('../Utilities/')
import printDecisionRegion as printDR
import matplotlib.pyplot as plt

print('Training a perceptron via scikit-learn')

# We can load Iris data from sklearn datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Loading Iris from sklearn')
print('X has ', len(X), ' Samples with ', len(X[0]), ' features')
print('y has ', len(y), ' Samples with unique value ', np.unique(y))

# Divide samples as Training(70%) and Testing(30%) 
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.3, random_state = 0)
print('Size after splitting data into Train and Test')
print('Train Sample Size: ', len(X_train))
print('Test Sample Size: ', len(X_test))

# Apply StandardScaler to fit train set
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 

# Now use perceptron to train the Train dataset and predict against 
# Test dataset to see accuracy
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('# of misclassfied samples ', (y_test != y_pred).sum())
print('Accuracy Score ', accuracy_score(y_test, y_pred))

# Print decision region and all samples
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
printDR.printDecisionRegionWithTestIdx(X = X_combined_std, y = y_combined, 
	                                   classifier = ppn, test_idx = range(105, 150))
plt.xlabel('petal length [Standarized]')
plt.ylabel('petal width [Standarized]')
plt.legend(loc = 'upper left')
plt.show()