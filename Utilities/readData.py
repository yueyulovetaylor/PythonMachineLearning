import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def readDataFromIris():
	# Use pandas library to import all data into a framework and 
	# tail the last five lines
	df = pd.read_csv('../Utilities/iris.data', header = None)

	print('Tail last five rows of the dataset')
	last5Samples = df.tail(5) 
	# tail(n) api returns the last n rows of the dataframe
	print(last5Samples)

	# Use the first 100 class labels with 50 1's (Versicolor) and
	# -1 (Setaso), also extract the first and third columns

	# Data clean and visualization
	y = df.iloc[0: 100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)
	X = df.iloc[0: 100, [0, 2]].values

	return {'X': X, 'y': y}

def readDataFromWine():
	df_wine = pd.read_csv('../Utilities/wine.data', header = None)
	df_wine.columns = ['Class label', 'Alcohol',
					   'Malic acid', 'Ash',
					   'Alcalinity of ash', 'Magnesium',
					   'Total phenols', 'Flavanoids',
					   'Nonflavanoid phenols',
					   'Proanthocyanins',
					   'Color intensity', 'Hue',
					   'OD280/OD315 of diluted wines',
					   'Proline']
	print('Class labels have values ', np.unique(df_wine['Class label']))
	print('Print first five rows of the dataset')
	print(df_wine.head())

	# Split data
	X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
	X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size = 0.3, random_state = 0)

	# Normalize the data
	mms = MinMaxScaler()
	X_train_norm = mms.fit_transform(X_train)
	X_test_norm = mms.transform(X_test)

	# Standardize the data
	stdsc = StandardScaler()
	X_train_std = stdsc.fit_transform(X_train),
	X_test_std = stdsc.transform(X_test)

	return {'X_train': X_train, 'y_train': y_train,
			'X_test': X_test, 'y_test':y_test,
			'X_train_norm': X_train_norm, 'X_test_norm': X_test_norm,
			'X_train_std': X_train_std, 'X_test_std': X_test_std,
			'columns': df_wine.columns}

def readDataFromWDBC():
	df_WDBC = pd.read_csv('../Utilities/wdbc.data', header = None)
	X = df_WDBC.loc[:, 2:].values
	y = df_WDBC.loc[:, 1].values
	le = LabelEncoder()
	y = le.fit_transform(y)

	print('After transform data [M, B] becomes %s' % le.transform(['M', 'B']))

	# Split the dataset into train and test with test accounts for 20%
	X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size = 0.2, random_state = 0)

	return {'X_train': X_train, 'y_train': y_train,
			'X_test': X_test, 'y_test':y_test}