# Use Sklearn.PCA and Logistic Regression to Predict Wine data
import sys
sys.path.append('../Utilities/')
import readData as RD
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import printDecisionRegion as printDR

DataMap = RD.readDataFromWine()
X_train_std = DataMap['X_train_std'][0]
X_test_std = DataMap['X_test_std']
y_train = DataMap['y_train']
y_test = DataMap['y_test']

# Contruct PCA and transform X
pca = PCA(n_components = 2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)

# Print the pca values
print('\nExplained Variance Ratio\n%s' % pca.explained_variance_ratio_)

# Print train Decision Region
print('Print Train Dataset Decision Region')
printDR.printDecisionRegion(X_train_pca, y_train, classifier = lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc = 'lower left')
plt.show()

# Print test Decision Region
print('Print Test Dataset Decision Region')
printDR.printDecisionRegion(X_test_pca, y_test, classifier = lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc = 'lower left')
plt.show()