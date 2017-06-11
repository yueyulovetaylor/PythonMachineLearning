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
		self.errors_ = []

		for nCt in range(self.n_iter):
			error = 0
			for xi, yi in zip(X, y):
				# Iterate through all input datasets to calculate error and update weight 
				# zip(*iterators) API, aggragates elements from each iterables
				update = self.eta * ( yi - self.labelZ(xi) )
				self.weight_[1:] += update * xi
				self.weight_[0] += update
				error += int(update != 0.0)
					# As long as an update is required, the prediction is incorrect
					# int API construct an integer from an boolean
			self.errors_.append(error)

		return self

	def calculateYHat(self, X):
		# Calculate YHat with current weight
		# use np.dot API, numpy.dot(a, b), dot product of two arrays
		return np.dot(X, self.weight_[1:]) + self.weight_[0]

	def labelZ(self, X):
		# For each input labelZ based on YHat
		# np.where API numpy.where(condition, x, y), condition can be arrayLike
		# return x if condition is met, y elsewise
		return np.where(self.calculateYHat(X) >= 0.0, 1, -1)
