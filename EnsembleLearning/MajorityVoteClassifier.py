# Implementation of Majority Voting Class
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVotingClassifier(BaseEstimator, ClassifierMixin):
	# Implementation of Majority Voting Classifier

	def __init__(self, classifiers, vote = 'classlabel', weights = None):
		# Constructor 
		self.classifiers = classifiers
		self.named_classifiers = {key: value for 
		                          key, value in _name_estimators(classifiers)}
		self.vote = vote
		self.weights = weights

	def fit(self, X, y):
		# Go through each classfier in the self.classifiers list
		# and get the training model against the <X, y> dataset
		self.labelnc_ = LabelEncoder()
		self.labelnc_.fit(y)
		self.classes_ = self.labelnc_.classes_
		self.classifiers_ = [] 
			# Private classifier that fit the input Trainint dataset

		for clf in self.classifiers:
			fitted_clf = clone(clf).fit(X, 
										self.labelnc_.transform(y))
			self.classifiers_.append(fitted_clf)
		return self

	def predict_proba(self, X):
		# This method calls the predict_proba of each input classifier
		# and get each classifier's sample lable probability
		# Then it takes the average
		probas = np.asarray([ clf.predict_proba(X) for clf in self.classifiers_ ])
		avg_proba = np.average(probas, axis = 0, weights = self.weights)
		return avg_proba

	def predict(self, X):
		# Predict y based on Probability and ClassLabel; however, 
		# since we only have 'classLabel' in this case, so here we 
		# only implement the ClassLabel case
		predictions = np.asarray([clf.predict(X) 
								  for clf in self.classifiers_]).T
			# Dimension of predictions is n_samples * n_classifiers
		maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights = self.weights)),
			      					   axis = 1,
			      					   arr = predictions)
			# major vote takes the label with largest probs and 
			# the size of the maj_vote is n_samples * 1
		maj_vot = self.labelnc_.inverse_transform(maj_vote)
			# Transform the major vote back to the original labels
		return maj_vote
