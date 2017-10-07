# Implementtion of AdaBoosting 
import sys
sys.path.append('../Utilities/')
import readData as RD

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

import numpy as np

print('Implementation of the AdaBoosting Algorithm')
print('1. Read the wine data')
DataMap = RD.readDataFromWine()
df_wine = DataMap['df_wine']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label']
X = df_wine[['Alcohol', 'Hue']].values

print("Number of samples we want to monitor is {0}".format(X.shape[0]))

print('\n2. Split the data into 60% Train and 40% Test')
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = \
	train_test_split(X, y,
					 test_size = 0.40,
					 random_state = 1)

print('\n3. Train the data with Tree and AdaBoosting Classifier')
tree = DecisionTreeClassifier(criterion = 'entropy',
							  max_depth = 1,	
							  	# In order to show the difference, depth needs to be 1
							  random_state = 0)	
ada = AdaBoostClassifier(base_estimator = tree,
						 n_estimators = 500,
						 learning_rate = 0.1,
						 random_state = 0)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision Tree train/test accuracies {0}/{1}'.format(round(tree_train, 3), 
														   round(tree_test, 3)))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('AdaBoost train/test accuracies {0}/{1}'.format(round(ada_train, 3), 
												  	  round(ada_test, 3)))

print('\n3. Print the Decision Region for both Decision Tree and AdaBoost')
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
				  np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows = 1, ncols = 2,
						sharex = 'col',
						sharey = 'row',
						figsize = (8, 3))

for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision Tree', 'AdaBoost']):
	clf.fit(X_train, y_train)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	axarr[idx].contourf(xx, yy, Z, alpha = 0.3)
	axarr[idx].scatter(X_train[y_train == 0, 0],
					   X_train[y_train == 0, 1],
					   c = 'blue',
					   marker = '^')
	axarr[idx].scatter(X_train[y_train == 1, 0],
					   X_train[y_train == 1, 1],
					   c = 'red',
					   marker = 'o')
	axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize = 12)
plt.text(10.2, -1.2,
		 s = 'Hue',
		 ha = 'center', va = 'center', fontsize = 12)

plt.show()