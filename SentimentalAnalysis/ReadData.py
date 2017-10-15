# Read ImDB Data into a csv file
import pyprind
import pandas as pd
import os
import numpy as np

pbar = pyprind.ProgBar(50000)  
	# ProgBar: simple progression bar for shell scripts
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

print('Read ImDB Data into a csv file')
print('1. Copy data into a DataFrame object')
for s in {'test', 'train'}:
	for l in {'pos', 'neg'}:
		path = './aclImdb/{0}/{1}'.format(s, l)
		for file in os.listdir(path):
			with open(os.path.join(path, file)) as infile:
				txt = infile.read()
				df = df.append([[txt, labels[l]]], ignore_index = True)
				pbar.update()

df.columns = ['review', 'sentiment']

print('2. Copy Data into a .csv file')
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index = False)

print('3. Print the first three rows of the csv')
df = pd.read_csv('./movie_data.csv')
print(df.head(3))