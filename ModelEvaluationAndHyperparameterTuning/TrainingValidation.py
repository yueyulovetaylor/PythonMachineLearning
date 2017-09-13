# Draw two curve, one is # of training numbers w/ Accuracy, the other is 
# parameter C w/ Accuracy
import sys
sys.path.append('../Utilities/')
import readData as RD

import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print('Draw Accuracy w/ # of training numbers and Parameter C')
print('1. Read WDBC data using Utility API')
DataMap = RD.readDataFromWDBC()
X_train = DataMap['X_train']
y_train = DataMap['y_train']
X_test = DataMap['X_test']
y_test = DataMap['y_test']

#Draw Accuracy w/ # of training numbers
print('\n2. Draw Accuracy w/ # of training numbers')
pipe_lr = Pipeline([('scl', StandardScaler()),
				    ('clf', LogisticRegression(
				    	penalty = 'l2', 
				    	random_state = 0)) ])
train_sizes, train_scores, test_scores = \
	learning_curve(estimator = pipe_lr,
				   X = X_train,
				   y = y_train,
				   train_sizes = np.linspace(0.1, 1.0, 10),
				   cv = 10,
				   n_jobs = 1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
print("Training statistic -- mean: %s; \nstd: %s" % (train_mean, train_std))

test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
print("Testing statistic -- mean: %s; \nstd: %s" % (test_mean, test_std))

# Plot the graph
plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', 
		 markersize = 5, label = 'Training Accuracy')
plt.fill_between(train_sizes,
				 train_mean + train_std,
				 train_mean - train_std,
				 alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color = 'green', linestyle = '--',
		 marker = 's', markersize = 5, label = 'Validation Accuracy')
plt.fill_between(train_sizes,
				 test_mean + test_std,
				 test_mean - test_std,
				 alpha = 0.15, color = 'green')
plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.8, 1.0])
plt.show()

# Draw Accuracy w/ parameter C
print('\n3. Draw Accuracy w/ Parameter C')
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
			estimator = pipe_lr,
			X = X_train,
			y = y_train,
			param_name = 'clf__C',
			param_range = param_range,
			cv = 10)

train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
print("Training statistic -- mean: %s; \nstd: %s" % (train_mean, train_std))

test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
print("Testing statistic -- mean: %s; \nstd: %s" % (test_mean, test_std))

# Plot the graph
plt.plot(param_range, train_mean, color = 'blue', marker = 'o', 
		 markersize = 5, label = 'Training Accuracy')
plt.fill_between(param_range,
				 train_mean + train_std,
				 train_mean - train_std,
				 alpha = 0.15, color = 'blue')
plt.plot(param_range, test_mean, color = 'green', linestyle = '--',
		 marker = 's', markersize = 5, label = 'Validation Accuracy')
plt.fill_between(param_range,
				 test_mean + test_std,
				 test_mean - test_std,
				 alpha = 0.15, color = 'green')
plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.8, 1.0])
plt.show()

print('\nConclusion: The Best C should be around 0.1')