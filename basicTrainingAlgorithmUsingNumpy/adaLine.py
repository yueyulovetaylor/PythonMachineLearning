# Implementation of adaLine learning algorithm
import numpy as np

class adaLineGD():
	def __init__(self, eta = 0.01, n_iter = 50):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		# Train dataset from a weight with all 0's through n_iter iterations
		# and learning rate as eta

		# Suppose X is a Matrix with m features (columns) and n samples (rows)
		self.weight_ = np.zeros(1 + X.shape[1]) 
		# cost_ array monitors each iterations cost
		self.cost_ = []

		for i in range(self.n_iter):
			# Go through all iterations
			output = self.net_input(X)
			errors = y - output # errors shape n * 1

			self.weight_[1: ] += self.eta * X.T.dot(errors)
				# X.T has size (m * n), errors has size (n * 1), with their 
				# multiplication, the size will be (m * 1), which is 
				# sum(error_i * x_i)
			self.weight_[0] += self.eta * errors.sum()
			curCost = (errors ** 2).sum() * 0.5
			self.cost_.append(curCost)

		return self

	def net_input(self, X):
		# Calculate the net value of the input, return's an array size 1*j
		# where j is # of samples
		# For 2-D arrays it is equivalent to matrix multiplication
		return np.dot(X, self.weight_[1:]) + self.weight_[0] 
			# Size X: n * m; size wieght weight m * 1, so final size (n * 1)

	def predict(self, X):
		# Return the classification value based on net_input result
		toClassify = self.net_input(X)
		return np.where(toClassify >= 0, 1, -1)

