# Implementation of Linear Regression Evaluation
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

print('Implementation of Linear Regression Evaluation')
print('1. Read the Housing Dataset')
df = pd.read_csv('./housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
			  'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
			  'B', 'LSTAT', 'MEDV']
print("Header of the DataFram: ")
print(df.head())
X = df.iloc[:, :-1].values
y = df['MEDV'].values

print('\n2. Train Test Split the dataset, with Test size as 30%')
print('   Train the data, and predict both Training and Test Samples')
X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size = 0.3, random_state = 0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print('\n3. Residual Plot for both Train and Test samples')
plt.scatter(y_train_pred, y_train_pred - y_train,
			c = 'blue', marker = 'o', label = 'Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test,
			c = 'lightgreen', marker = 's', label = 'Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
plt.show()

print('\n4. Mean Squared Error (MSE) Result')
print('MSE_Train: {0}'.format(round(mean_squared_error(y_train, y_train_pred), 3)))
print('MSE_Test: {0}'.format(round(mean_squared_error(y_test, y_test_pred), 3)))

print('\n5. R square Result')
print('R Square Train: {0}'.format(round(r2_score(y_train, y_train_pred), 3)))
print('R Square Test: {0}'.format(round(r2_score(y_test, y_test_pred), 3)))