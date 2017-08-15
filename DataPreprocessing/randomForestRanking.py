# Use Random Forest to rank features
import sys
sys.path.append('../Utilities/')
import readData as RD

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# Read data out
DataMap = RD.readDataFromWine()
X_train = DataMap['X_train']
y_train = DataMap['y_train']
feature_labels = DataMap['columns']

# Train the forest model
forest = RandomForestClassifier(n_estimators = 10000,
	                            random_state = 0,
	                            n_jobs = -1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
	print("%2d) %-*s %f" % (f + 1, 
							30, 
							feature_labels[indices[f]], 
							importances[f]))

# Plot feature importances
plt.title('Features Importances')
plt.bar(range(X_train.shape[1]), 
		importances[indices],
	    color = 'lightblue',
	    align = 'center')
plt.xticks(range(X_train.shape[1]),
	       feature_labels[indices],
	       rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()