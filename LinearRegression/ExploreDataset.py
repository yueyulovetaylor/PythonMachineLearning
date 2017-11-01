# Implementation of Exploring the Housing Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print('Explore the Housing Dataset')
print('1. Read the Housing Dataset')
df = pd.read_csv('./housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
			  'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
			  'B', 'LSTAT', 'MEDV']
print("Header of the DataFram: ")
print(df.head())

print('\n2. Visualize pair-wise scatterplot matrix')
sns.set(style = 'whitegrid', context = 'notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size = 1.5)
plt.show()

print('\n3. Plot the Correlation Matrix Heatmap')
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,
				 cbar = True,
				 annot = True,
				 square = True,
				 fmt = '.2f',
				 annot_kws = {'size': 15},
				 yticklabels = cols,
				 xticklabels = cols)
plt.show()