# Implementation of Perceptron Learning, 
# Content in Chapter 2 of Python Machine learning
import numpy as np

class Perceptron():

	def __init__(self, eta = 0.01, n_iter = 10):
		# Contructor, learning rate initialized as 0.01; 
		# # of iterations initialized as 10
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		# Train dataset from a weight with all 0's through n_iter iterations
		# and learning rate as eta
		self.weight_ = np.zeros(1 + X.shape[1]) 
			# Weight has one more dimension than X, represention a constant offset 
			# from the model
		self.error = []

		for _ in range(self.n_iter):
			error = 0
			for xi, yBar in zip(X, y):
				# Iterate through all input datasets to calculate error and update weight 
				# zip(*iterators) API, aggragates elements from each iterables

	def calculateYHat(self, X):
		# Calculate YHat with current weight
		# use np.dot API, numpy.dot(a, b), dot product of two arrays
		return np.dot(X, self.w_[1:]) + w[0]

	def labelZ(self, X):
		# For each input labelZ based on YHat
		# np.where API numpy.where(condition, x, y), condition can be arrayLike
		# return x if condition is met, y elsewise
		return np.where(self.calculateYHat(X), 1, -1)