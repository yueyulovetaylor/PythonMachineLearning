from perceptronLearning import Perceptron
import sys
sys.path.append('Utilities/')
import printDecisionRegion as printDR
import readData as RD

import matplotlib.pyplot as plt
import numpy as np

print('Implementation of Perceptron Learning')

X = RD.readDataFromIris()['X']
y = RD.readDataFromIris()['y']

# Plot red 'o' as Setosa and 'x' as 'Versicolor'
plt.scatter(X[:50, 0], X[0: 50, 1], color = 'red', marker = 'o', label = 'Setaso')
plt.scatter(X[50: 100, 0], X[50: 100, 1], color = 'blue', marker = 'x', label = 'Virginica')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
print('\nPlot Initial dataset in two dimensions')
plt.show()

# Use perceptron to train the data for classification
ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

# Plot the errors
print('Errors by the end of each iteration are')
print(ppn.errors_)
print('Plot error by iteration')
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassification')
plt.show()

# Plot Decision Region
printDR.printDecisionRegion(X, y, ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()
