# Apply Implemented Majority Vote Classifier to the Iris Data
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from itertools import product
from sklearn.grid_search import GridSearchCV

from MajorityVoteClassifier import MajorityVotingClassifier

print('Apply Implemented Majority Vote Classifier to the Iris Data')
print('1. Read Iris Data from sklearn.datasets')
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
print('Sample Size {0}'.format(len(y)))
le = LabelEncoder()
y = le.fit_transform(y)
print('Split the sample into 50% train and 50% test')
X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size = 0.5, random_state = 1)

print('\n2. Before Building Majority Vote Classifier, Establish Three Classifiers')
print('Evaluate the performance of each one of them')
print('Individual Classifiers include Logistic Regression, Decision Tree and K-Nearest Neighbor')
clf1 = LogisticRegression(penalty = 'l2', C = 0.001, random_state = 0)
clf2 = DecisionTreeClassifier(max_depth = 1, criterion = 'entropy', random_state = 0)
clf3 = KNeighborsClassifier(n_neighbors = 1, p = 2, metric = 'minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],
	              ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
				  ['clf', clf3]])
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('\n  Use 10-fold cross validation to evaluate each classifier\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
	scores = cross_val_score(estimator = clf,
							 X = X_train,
							 y = y_train,
							 cv = 10,
							 scoring = 'roc_auc')
	print("ROC AUC: {0} (+/- {1}) [{2}]".format(
		round(scores.mean(), 2), round(scores.std(), 2), label))

print('\n3. Now we establish the Majority Voting Classifier and compare its performance with others')
mv_clf = MajorityVotingClassifier(classifiers = [pipe1, clf2, pipe3])
scores = cross_val_score(estimator = mv_clf,
						 X = X_train,
						 y = y_train,
						 cv = 10,
						 scoring = 'roc_auc')
print('Use 10-fold cross_validation to evaluate Majority Voting')
print("ROC AUC: {0} (+/- {1}) [Majority Voting]".format(
	  round(scores.mean(), 2), round(scores.std(), 2)))

print('\n4. Plotting: ROC Curve and Decision Region')
print('In this section, we will plot the ROC Curve and Decision Region for all four classfiers')

print('\n4.1 ROC Curve Plotting')
all_clf = [pipe1, clf2, pipe3, mv_clf]
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
clf_labels.append('Majority Voting')
for clf, label, clr, ls \
		in zip(all_clf, clf_labels, colors, linestyles):
	y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
	fpr, tpr, threshols = roc_curve(y_true = y_test, y_score = y_pred)
	roc_auc = auc(x = fpr, y = tpr)
	plt.plot(fpr, tpr, color = clr, linestyle = ls, 
			 label = "{0} (auc = {1})".format(label, round(roc_auc, 2)))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', linewidth = 2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print('\n4.2 Decision Region Plotting')
print('Standard Scale the datasets')
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
print('On Coordination, <X_min, X_max> is <{0}, {1}>'.format(
	  round(X_train_std[:, 0].min(), 4), 
	  round(X_train_std[:, 0].max(), 4)))

y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
print('On Coordination, <Y_min, Y_max> is <{0}, {1}>'.format(
	  round(X_train_std[:, 1].min(), 4), 
	  round(X_train_std[:, 1].max(), 4)))

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
					 np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows = 2, ncols = 2,
					   sharex = 'col',
					   sharey = 'row',
					   figsize = (7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
	clf.fit(X_train_std, y_train)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	axarr[idx[0], idx[1]].contourf(xx, yy ,Z, alpha = 0.3)
	axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
								  X_train_std[y_train == 0, 1],
								  c = 'blue',
								  marker = '^',
								  s = 50)
	axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
								  X_train_std[y_train == 1, 1],
								  c = 'red',
								  marker = 'o',
								  s = 50)
	axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -4.5, 
		 s = 'Sepal Width (Standard Scaled)',
		 ha = 'center', va = 'center', fontsize = 12)
plt.text(-10.5, 4.5, 
		 s = 'Petal Length (Standard Scaled)',
		 ha = 'center', va = 'center', fontsize = 12, rotation = 90)

plt.show()

print('\n5. Print All Parameters')
print('5.1 Dump Useful Model Parameters for Grid Search Reference')
all_param = mv_clf.get_params()

print('Print all parameters of Logistic Regression and Decision')
print('\n5.1 Logistic Regression')
lr_param = all_param['classifiers'][0].steps[1][1]
print(lr_param)

print('\n5.2 Decision Tree')
DT_param = all_param['classifiers'][1]
print(DT_param)
