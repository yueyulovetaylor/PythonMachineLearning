import sys
sys.path.append('../Utilities/')
import readData as RD
import printDecisionRegion as printDR
import numpy as np
import matplotlib.pyplot as plt

from adaLineSGD import adaLineSGD

print('Implementation of Ada Line using Stochastic Gradient Descent')

DataMap = RD.readDataFromIris()
X = DataMap['X']
y = DataMap['y']

# We standardize X using (x_j - mu_j) / std_j
print('standardize X using (x_j - mu_j) / std_j')
X_std = np.copy(X)
X_std[:, 0] = ( X[:, 0] - X[:, 0].mean() ) / X[:, 0].std()
X_std[:, 1] = ( X[:, 1] - X[:, 1].mean() ) / X[:, 1].std()

adaSGD = adaLineSGD(n_iter = 15, eta = 0.01, random_state = 1)
adaSGD.fit(X_std, y)

# Get the cost weight increase of epchos
print('Cost of AdaSGD learning with the increase of epchos ', adaSGD.cost_)
print('PlotCost of AdaSGD learning with the increase of epchos')
plt.plot(range(1, len(adaSGD.cost_) + 1), adaSGD.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# Plot the decision region
printDR.printDecisionRegion(X_std, y, classifier = adaSGD)
plt.xlabel('sepal length [Standardized]')
plt.ylabel('petal length [Standardized]')
plt.title('Adaline - Stochastic Gradient Descent')
plt.legend(loc = 'upper left')
plt.show()