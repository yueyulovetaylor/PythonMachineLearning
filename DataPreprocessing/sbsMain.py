# Utilize SBS algorithm into real data sample
import sys
sys.path.append('../Utilities/')
import readData as RD
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sbs import SBS

# Read data out
DataMap = RD.readDataFromWine()
X_train_std = DataMap['X_train_std'][0]
y_train = DataMap['y_train']
print(X_train_std)
X_test_std = DataMap['X_test_std']
y_test = DataMap['y_test']
print(y_test.shape, X_test_std.shape)
print(X_test_std)
columnsLabels = DataMap['columns']

# Fit the data with SBS
knn = KNeighborsClassifier(n_neighbors = 2)
sbs = SBS(knn, k_features = 1)
sbs.fit(X_train_std, y_train)

# Plot the accuracy tendency
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker = 'o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# Outprint the chosen features
print('5 features has the best outcome')
k5 = list(sbs.subsets_[8])
print('features select', columnsLabels[1:][k5])

# Train the original dataset
print('Train Original Dataset')
knn.fit(X_train_std, y_train)
print('Train Accuracy', knn.score(X_train_std, y_train))
print('Test Accuracy', knn.score(X_test_std, y_test))

# Train with selected features
print('Train with selected 5 features')
knn.fit(X_train_std[:, k5], y_train)
print('Train Accuracy', knn.score(X_train_std[:, k5], y_train))
print('Test Accuracy', knn.score(X_test_std[:, k5], y_test))