# Implement Logistic Regression using L1 Regularization and 
# compare different C scenarios
import sys
sys.path.append('../Utilities/')
import readData as RD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import math

DataMap = RD.readDataFromWine()
X_train_std = DataMap['X_train_std']
y_train = DataMap['y_train']
columnsLabels = DataMap['columns']

# Get the background before plotting
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []

# Train with different Penalization Parameters
for c in np.arange(-4, 6):
	input_C = math.pow(10, c)
	lr = LogisticRegression(penalty = 'l1', C = input_C, random_state = 0)
	lr.fit(X_train_std[0], y_train)
	weights.append(lr.coef_[1])
	params.append(input_C)

weights = np.array(weights)
print('weights array is ', weights)

for column, color in zip(range(weights.shape[1]), colors):
	plt.plot(params, weights[:, column], 
			 label = columnsLabels[column + 1], color = color)

plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
plt.xlim([math.pow(10, -5), math.pow(10, 5)])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc = 'upper center', bbox_to_anchor = (1.38, 1.03), ncol = 1, fancybox = True)

plt.show()