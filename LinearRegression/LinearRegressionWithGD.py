# Implementation of Linear Regression using Gradient Descent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

class LinearRegressionGD(object):
	def __init__(self, eta = 0.001, n_iter = 20):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		self.w_ = np.zeros(1 + X.shape[1])	# Initialize all weight as 0
		self.cost_ = []

		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors ** 2).sum() / 2.0
			self.cost_.append(cost)
		return self

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		return self.net_input(X)

def lin_regplot(X, y, model):
	plt.scatter(X, y, c = 'blue')
	plt.plot(X, model.predict(X), color = 'red')
	return None

print('Implementation of Linear Regression using Gradient Descent')
print('1. Read the Housing Dataset')
df = pd.read_csv('./housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
			  'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
			  'B', 'LSTAT', 'MEDV']
print("Header of the DataFram: ")
print(df.head())

print('\n2. Use the Customized Linear Regression GD Object to fit the model')
X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

print('Plot SSE (Cost Function Value) vs. Epoch')
plt.plot(range(1, lr.n_iter + 1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

print('Plot Real vs. Prediction Graph')
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average # of rooms [RM] Standardized')
plt.ylabel('Price in $1000\'s [MEDV] Standardized')
plt.show()

print('Prediction of 5.0 Room Apartments')
num_rooms_std = sc_x.transform([5.0])
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: {0}".format(round(sc_y.inverse_transform(price_std), 3)))

print("Parameters -- Slope: {0}; Intercept: {1}".format(round(lr.w_[1], 3),
														round(lr.w_[0], 3)))

print('3. Use Linear Regression Library from sklearn')
slr = LinearRegression()
slr.fit(X, y)
print("Parameters -- Slope: {0}; Intercept: {1}".format(round(slr.coef_[0], 3),
														round(slr.intercept_, 3)))

lin_regplot(X, y, slr)
plt.xlabel('Average # of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()