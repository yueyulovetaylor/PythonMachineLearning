import pandas as pd
import numpy as np

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
