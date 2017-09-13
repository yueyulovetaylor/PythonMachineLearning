# Implementation of using Pipeline object to integrate a learning algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import sys
sys.path.append('../Utilities/')
import readData as RD

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

print('Use Pipeline to combine transformers and estimators')
print('1. Read WDBC data using Utility API')
DataMap = RD.readDataFromWDBC()
X_train = DataMap['X_train']
y_train = DataMap['y_train']
X_test = DataMap['X_test']
y_test = DataMap['y_test']

print('\n2. Integrate a LogisticRegression into a Pipeline')
pipe_lr = Pipeline([('scl', StandardScaler()),
					('pca', PCA(n_components = 2)),
					('clf', LogisticRegression(random_state = 1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

print('\n3. Manually implement K-fold validation')
kfold = StratifiedKFold(y = y_train,
					    n_folds = 10,
					    random_state = 1)
scores = []
print('Go through each iteration to calculate its Prediction Score')
print('Note: ny.bincount -- ' + 
	  '\nCount number of occurrences of each value in array of non-negative ints.\n')
for k, (train, test) in enumerate(kfold):
	pipe_lr.fit(X_train[train], y_train[train])
	score = pipe_lr.score(X_train[test], y_train[test])
	scores.append(score)
	print('Fold: %s, class dist.: %s, Acc: %.3f' % 
		  (k + 1, np.bincount(y_train[train]), score))

print('\n=> Final Result:')
print('CV accuracy: %.3f +/- %.3f' %
	  (np.mean(scores), np.std(scores)))

print('\n4. Compare result in section 3 with calling sklearn cv library')
scores = cross_val_score(estimator = pipe_lr,
						 X = X_train,
						 y = y_train,
						 cv = 10,
						 n_jobs = 1)
print('CV accuracy scores %s' % scores)
print('=> Final result')
print('CV accuracy: %.3f +/- %.3f' %
	  (np.mean(scores), np.std(scores)))