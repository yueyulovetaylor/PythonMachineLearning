# Implementation of Grid Search to find best parameter in the model
import sys
sys.path.append('../Utilities/')
import readData as RD

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print('Implementation of Grid Search to find best parameter in the model')
print('1. Read WDBC data using Utility API')
DataMap = RD.readDataFromWDBC()
X_train = DataMap['X_train']
y_train = DataMap['y_train']
X_test = DataMap['X_test']
y_test = DataMap['y_test']

print('\n2. Use Grid Search to Find the best parameter in the SVM model')
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
	              cv = 10,
	              n_jobs = -1)
gs = gs.fit(X_train, y_train)
print('The Best Score we can get is %.4f' % gs.best_score_)
print('The Best Parameters are:')
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('\nUse Best Estimator to train data result is %.4f' 
	  % clf.score(X_test, y_test))