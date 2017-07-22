# Plot the each of the three impurities
import matplotlib.pyplot as plt
import numpy as np

# Utility function to calculate impurities; 
# Assume all splits are binary 
def gini(p):
	return 2 * p * (1 - p)

def entropy(p):
	return -p * np.log2(p) - (1-p) * np.log2(1-p)

def error(p):
	return 1 - np.max([p, 1-p])

print('Print each of the three impurities in a single graph')
X = np.arange(0.0, 1.0, 0.01)

# Calculate the entropy
ent = [entropy(p) if p != 0 else None for p in X]
sc_ent = [e*0.5 if e else None for e in ent]
print('Scaled Entropy in an array is', sc_ent)

err = [error(i) for i in X]
print('Misclassification in an array is', err)

# Plot all of four Impurities
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(X), err], 
	['Entropy', 'Entropy (Scaled)', 'Gini Impurity', 'Misclassification Error'],
	['-', '-', '--', '-.'],
	['black', 'lightgray', 'red', 'green', 'cyan']):
	line = ax.plot(X, i, label = lab, linestyle = ls, lw = 2, color = c)
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 0.15), 
	ncol = 3, fancybox = True, shadow = False)
ax.axhline(y = 0.5, linewidth = 1, color = 'k', linestyle = '--')
ax.axhline(y = 1.0, linewidth = 1, color = 'k', linestyle = '--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.title('Plot of each Impurities with the increase of probability')
plt.show()