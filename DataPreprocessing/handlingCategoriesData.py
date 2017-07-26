# How to handle category data
import pandas as pd
import numpy as np

# Input categorical data
df = pd.DataFrame([
	['green', 'M', 10.1, 'class1'],
	['red', 'L', 13.5, 'class2'],
	['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classLabel']
print('Input Categorical Data Frame')
print(df)

# Map Size
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)

# Map classLabel
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classLabel']))}
df['classLabel'] = df['classLabel'].map(class_mapping)

# One Hot Encoding on Color
df = pd.get_dummies(df[['price', 'color', 'size']])

print('After mapping all fields we get')
print(df)