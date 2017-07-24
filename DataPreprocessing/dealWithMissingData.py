# How to deal with Missing Data
import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

# Read csv data into a data frame
csv_data = '''A, B, C, D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
print("Read from CSV data into data frame")
print(df)

print('Whether each column has a missing field')
print(df.isnull().sum())

# Imputing missing values
imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print('Print the dataset after imputing')
print(imputed_data)
