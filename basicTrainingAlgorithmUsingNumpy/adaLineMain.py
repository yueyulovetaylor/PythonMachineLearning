import sys
sys.path.append('../Utilities/')
import readData as RD
import printDecisionRegion as printDR
import numpy as np
import matplotlib.pyplot as plt

from adaLine import adaLineGD

print('Implementation of AdaLine Learning')

DataMap = RD.readDataFromIris()
X = DataMap['X']
y = DataMap['y']

# Compare to AdaLine learning by changing eta between 10e-2 and 10e-4
print('construct two adaLine model with eta 0.01 and 0.0001')
ada1 = adaLineGD(eta = 0.01, n_iter = 10).fit(X, y)
ada2 = adaLineGD(eta = 0.0001, n_iter = 10).fit(X, y)

print("Ada1 costs are ", np.log10(ada1.cost_))
print("Ada2 costs are ", ada2.cost_)

print('Plot change of cost or log10(cost) with the increase of number of iteration')

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4)) 
	# divide into a (1 * 2) figures plot
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('ada-line with learning rate 0.01')

ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('sum-squared-error')
ax[1].set_title('ada-line with learning rate 0.0001')

plt.show()

# We standardize X using (x_j - mu_j) / std_j
print('standardize X using (x_j - mu_j) / std_j')
X_std = np.copy(X)
X_std[:, 0] = ( X[:, 0] - X[:, 0].mean() ) / X[:, 0].std()
X_std[:, 1] = ( X[:, 1] - X[:, 1].mean() ) / X[:, 1].std()

adaStd = adaLineGD(n_iter = 15, eta = 0.01)
adaStd.fit(X_std, y)
print('Cost of X std ada learning', adaStd.cost_)

# Plot cost with epochs first
plt.plot(range(1, len(adaStd.cost_) + 1), adaStd.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('sum-squared-error')
plt.title('Standardized ada-line with learning rate 0.01')
plt.show()

# Finally, we plot the decision region
print("Plot Decision Region")
printDR.printDecisionRegion(X_std, y, classifier = adaStd)
plt.xlabel('sepal length [Standardized]')
plt.ylabel('petal length [Standardized]')
plt.title('Adaline - Gradient Descent')
plt.legend(loc = 'upper left')
plt.show()

