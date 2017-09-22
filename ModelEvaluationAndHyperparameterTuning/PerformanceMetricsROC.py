# Implementations of the following several fields regarding Performance Metrics
# 1. Confusion Matrix
# 2. Calculate Precision, Recall and F1
# 3. Plot ROC

import sys
sys.path.append('../Utilities/')
import readData as RD

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from scipy import interp
import numpy as np

print('Implementation of Performace Metrics related fields')
print('1. Read WDBC data using Utility API')
DataMap = RD.readDataFromWDBC()
X_train = DataMap['X_train']
y_train = DataMap['y_train']
X_test = DataMap['X_test']
y_test = DataMap['y_test']

print('\n2. Calculate and plot the confusion matrix')
pipe_svc = Pipeline([('scl', StandardScaler()),
					 ('clf', SVC(random_state = 1))])
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print('After calculation, Confusion Matrix is')
print(confmat)
print('Then, we plot Confusion Matrix in the following graph')
fig, ax = plt.subplots(figsize = (2.5, 2.5))
ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)

for i in range(confmat.shape[0]):
	for j in range(confmat.shape[1]):
		ax.text(x = j, y = i,
			    s = confmat[i, j],
			    va = 'center',
			    ha = 'center')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print('\n3. Precision, Recall and F1')
print('Precision of the Test Samples is %.3f' % 
	  precision_score(y_true = y_test, y_pred = y_pred))
print('Recall of the Recall Samples is %.3f' % 
	  recall_score(y_true = y_test, y_pred = y_pred))
print('F1 Score of the Recall Samples is %.3f' % 
	  f1_score(y_true = y_test, y_pred = y_pred))

print('\n4. Plot ROC Curve')
pipe_lr = Pipeline([('scl', StandardScaler()),
	                ('pca', PCA(n_components = 2)),
	                ('clf', LogisticRegression(penalty = 'l2',
	                						   random_state = 0,
	                						   C = 100.0))])
print('After establish the pipeline, we only take two features \
from the Breast Cancer Dataset')
X_train2 = X_train[:, [4, 14]]
cv = StratifiedKFold(y_train, n_folds = 3, random_state = 1)
fig = plt.figure(figsize = (7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

print('Now go through each CV Dataset to establish the ROC based FPR \
and TPR of each')
for i, (train, test) in enumerate(cv):
	print('Begin to Process CV Class {0}.'.format(i))
	probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(
		X_train2[test])
		# The return can be interpreted as (# of samples) * (# of classes)
		# Each matrix element can is the possibility of Sample_i's probability
		# as Class j
	fpr, tpr, threshold = roc_curve(y_train[test],
									probas[:, 1], 
									pos_label = 1)
		# For each i in all three arrays, fpr[i] and tpr[i] are the corresponding
		# value when score > threshold[i]
	mean_tpr += interp(mean_fpr, fpr, tpr)
	mean_tpr[0] = 0.0
	roc_auc = auc(fpr, tpr)
	print('CV Class {0} has ROC_AUC_score = {1}\n'.format(i+1, round(roc_auc, 3)))
	plt.plot(fpr, tpr, lw = 1,
			 label = 'ROC fold {0} (area = {1})'.format(i+1, round(roc_auc, 3)))

# Plot the random guess line, which is a straight line from (0,0) to (1,1)
plt.plot([0, 1],
		 [0, 1],
		 linestyle = '--',
		 color = (0.6, 0.6, 0.6),
		 label = 'random guessing')

# Plot the mean ROC line and compute its AUC_score
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0    # Mark the last item in the array as 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
		 label = 'Mean ROC (area = {0})'.format(round(mean_auc, 3)),
		 lw = 2)

# Plot the perfect performance
plt.plot([0, 0, 1],
	     [0, 1, 1],
	     lw = 2,
	     linestyle = ':',
	     color = 'black',
	     label = 'Perfect Performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic')

plt.legend(loc = 'lower right')
plt.show()

print('\n5. Run against Test Data')
pipe_lr = pipe_lr.fit(X_train2, y_train)
y_pred2 = pipe_lr.predict(X_test[:, [4, 14]])
print('ROC AUC score = {0}'.format(round(roc_auc_score(
		y_true = y_test, y_score = y_pred2), 3)))

print('Accuracy score = {0}'.format(round(accuracy_score(
		y_true = y_test, y_pred = y_pred2), 3)))
