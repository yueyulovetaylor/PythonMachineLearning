# Implementation of Polynomial Regression using the House Data
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

print('Implementation of Polynomial Regression using the House Data')
print('1. Read the Housing Dataset')
df = pd.read_csv('./housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
			  'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
			  'B', 'LSTAT', 'MEDV']
print("Header of the DataFram: ")
print(df.head())

print('\n2. Create Quadratic and Cubic features')
X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()
quadratic = PolynomialFeatures(degree = 2)
cubic = PolynomialFeatures(degree = 3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

print('\n3. Linear Fit, Quadratic Fit and Cubic Fit')
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

print('\n4. Plot all results')
plt.scatter(X, y,
			label = 'training points',
			color = 'lightgray')
plt.plot(X_fit, y_lin_fit,
		 label = 'Linear (d = 1), R Square = {0}'.format(round(linear_r2, 3)),
		 color = 'blue',
		 lw = 2,
		 linestyle = ':')
plt.plot(X_fit, y_quad_fit,
		 label = 'Quadratic (d = 2), R Square = {0}'.format(round(quadratic_r2, 3)),
		 color = 'red',
		 lw = 2,
		 linestyle = '-')
plt.plot(X_fit, y_cubic_fit,
		 label = 'Cubic (d = 3), R Square = {0}'.format(round(cubic_r2, 3)),
		 color = 'green',
		 lw = 2,
		 linestyle = '--')
plt.xlabel("Percentage lower status of the population [LSTAT]")
plt.ylabel("Price in $1000\'s [MEDV]")
plt.legend(loc = 'upper right')
plt.show()

print('\n5. Plot y_sqrt vs. X_log relationship, Linear Regression')
X_log = np.log(X)
y_sqrt = np.sqrt(y)

X_fit = np.arange(X_log.min() - 1,
				  X_log.max() + 1,
				  1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

plt.scatter(X_log, y_sqrt,
			label = 'Training Points',
			color = 'lightgray')
plt.plot(X_fit, y_lin_fit,
		 label = 'Linear (d = 1) R Square = {0}'.format(round(linear_r2, 3)),
		 color = 'blue',
		 lw = 2)
plt.xlabel('log(Percentage lower status of the population [LSTAT])')
plt.ylabel('sqrt(Price in $1000\'s [MEDV])')
plt.legend(loc = 'lower left')
plt.show()