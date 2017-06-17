# Implementation of Ada Line Learning with Stochastic Gradient Descent
import numpy as np
from numpy.random import seed

class adaLineSGD():
	# Ada Line using Stochastic Gradient Descent
	def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle

		if (random_state):
			seed(random_state)

	def fit(self, X, y):
		self._initialize_weight(X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffule(X, y)
			cost = []

			for xi, target in zip(X, y):
				# Go through each sample to update weights
				cost.append(self._update_weights(xi, target))

			avg_cost = sum(cost) / len(y)
			self.cost_.append(avg_cost)

		return self

	def _initialize_weight(self, m):
		# Initialize weights to 0, m is the number of features
		self.w_ = np.zeros(m + 1)
		self.w_initialized = False

	def _shuffule(self, X, y):
		# Shuttle the training data
		r = np.random.permutation(len(y))
			# Return a permuted range
		return X[r], y[r]

	def _update_weights(self, xi, target):
		# Apply Adaline Learning rule to one sample in order to update the weight
		output_i = self.net_input(xi)
		error_i = target - output_i
		self.w_[1: ] += self.eta * xi.dot(error_i)
		self.w_[0] += self.eta * error_i

		cost_i = 0.5 * (error_i ** 2)
		return cost_i

	def net_input(self, X):
		# Calculate net input's linear value
		return np.dot(X, self.w_[1: ]) + self.w_[0]

	def predict(self, X):
		# Return the classification value based on net_input result
		toClassify = self.net_input(X)
		return np.where(toClassify >= 0, 1, -1)


