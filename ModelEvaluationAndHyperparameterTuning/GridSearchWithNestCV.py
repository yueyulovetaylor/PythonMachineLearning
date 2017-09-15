# Implementation of Cross Validation to Grid Search for best parameter
import sys
sys.path.append('../Utilities/')
import readData as RD

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

print('Implementation of Cross Validation to Grid Search for best parameter')
print('1. Read WDBC data using Utility API')
DataMap = RD.readDataFromWDBC()
X_train = DataMap['X_train']
y_train = DataMap['y_train']
X_test = DataMap['X_test']
y_test = DataMap['y_test']

print('\n2. Initiate the Grid Search CV object')
pipe_svc = Pipeline([('scl', StandardScaler()),
					 ('clf', SVC(random_state = 1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
			   'clf__kernel': ['linear']}, 
			  {'clf__C': param_range,
			   'clf__gamma': param_range,
			   'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator = pipe_svc,
	              param_grid = param_grid,
	              scoring = 'accuracy',
	              cv = 2,
	              n_jobs = -1)
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy',
	                     cv = 5)
print('CV accuracy: %.3f +/- %.3f' % 
	  (np.mean(scores), np.std(scores)))

print('\n3. Use Decision Tree Classifier to again do CV and tuning the parameter')
gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 0),
				  param_grid = [
				      {'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
				  scoring = 'accuracy',
				  cv = 5)
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 2)
print('CV accuracy: %.3f +/- %.3f' % 
	  (np.mean(scores), np.std(scores)))