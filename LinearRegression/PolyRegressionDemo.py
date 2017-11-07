# Implementation of a Demo for Polynomial Regression
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

print('Implementation of a Demo for Polynomial Regression')
print('1. Demo Samples')
X = np.array([258.0, 270.0, 294.0,
			  320.0, 342.0, 368.0,
			  396.0, 446.0, 480.0,
			  586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8,
			  298.6, 314.2, 342.2,
			  360.8, 368.0, 391.2,
			  390.8])
print('Sample size is {0}'.format(y.shape[0]))
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree = 2)
X_quad = quadratic.fit_transform(X)

print('\n2. Linear Regression (Simple Linear Regression and Polynomial)')
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

print('\n3. Plot the results')
plt.scatter(X, y, label = 'training points')
plt.plot(X_fit, y_lin_fit,
		 label = 'linear fit', linestyle = '--')
plt.plot(X_fit, y_quad_fit,
		 label = 'quadratic')
plt.legend(loc = 'upper left')
plt.show()

print('\n4. Performance Evaluation')
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)

print('Training MSE Linear: {0}; Quadratic: {1}'.format(
		round(mean_squared_error(y, y_lin_pred), 3),
		round(mean_squared_error(y, y_quad_pred), 3)))

print('Training R Sqaure Linear: {0}; Quadratic: {1}'.format(
		round(r2_score(y, y_lin_pred), 3),
		round(r2_score(y, y_quad_pred), 3)))
