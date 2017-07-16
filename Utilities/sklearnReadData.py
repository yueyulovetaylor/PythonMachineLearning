# Use sklearn library to fetch iris data
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

def sklearnReadIris():
	# We can load Iris data from sklearn datasets
	iris = datasets.load_iris()
	X = iris.data[:, [2, 3]]
	y = iris.target
	print('Loading Iris from sklearn')
	print('X has ', len(X), ' Samples with ', len(X[0]), ' features')
	print('y has ', len(y), ' Samples with unique value ', np.unique(y))

	# Divide samples as Training(70%) and Testing(30%) 
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size = 0.3, random_state = 0)
	print('Size after splitting data into Train and Test')
	print('Train Sample Size: ', len(X_train))
	print('Test Sample Size: ', len(X_test))

	# Apply StandardScaler to fit train set
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test) 

	return {'X_train_std': X_train_std, 'y_train': y_train, 
	        'X_test_std': X_test_std, 'y_test': y_test}