# Implementation for both Decision Tree and Random Forest Regression
# algorithm
import warnings
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def lin_regplot(X, y, model):
	plt.scatter(X, y, c = 'blue')
	plt.plot(X, model.predict(X), color = 'red')
	return None

print('Implementation of Decision Tree and Random Forest Regression')
print('1. Read the Housing Dataset')
df = pd.read_csv('./housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
			  'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
			  'B', 'LSTAT', 'MEDV']
print("Header of the DataFram: ")
print(df.head())

print('\n2. Decision Tree Regression: LSTAT as X; MEDV as Y')
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth = 3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel("Percentage of lower startus of the population [LSTAT]")
plt.ylabel('Price in 1000 USD [MEDV]')
plt.show()

print('\n3. Random Forest Regression: all other(X) vs. MEDV(Y)')
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = \
	train_test_split(X, y,
					 test_size = 0.4,
					 random_state = 1)
forest = RandomForestRegressor(n_estimators = 1000,
							   criterion = 'mse',
							   random_state = 1,
							   n_jobs = -1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('3.1 MSE and R-Square Scores')
print('MSE Train: {0}, Test: {1}'.format(
		round(mean_squared_error(y_train, y_train_pred), 3),
		round(mean_squared_error(y_test, y_test_pred), 3)))
print('R-Square Train: {0}, Test: {1}'.format(
		round(r2_score(y_train, y_train_pred), 3),
		round(r2_score(y_test, y_test_pred), 3)))

print('\n3.2 Plot Residues vs. Prediction Values')
plt.scatter(y_train_pred,
			y_train_pred - y_train,
			c = 'black',
			marker = 'o',
			s = 35, 
			alpha = 0.5,
			label = 'Training Data')
plt.scatter(y_test_pred,
			y_test_pred - y_test,
			c = 'lightgreen',
			marker = 's',
			s = 35,
			alpha = 0.7,
			label = 'Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
plt.show()
