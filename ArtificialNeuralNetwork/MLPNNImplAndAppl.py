# Implementation of MLP Neural Network Object (Class)
# Application of ANN to Hand Written Digits' Data
import numpy as np
from scipy.special import expit
import sys

# Implementations of MLP Neural Network Object
class NeuralNetMLP(object):
	def __init__(self, n_output, n_features, n_hidden = 30,
				 l1 = 0.0, l2 = 0.0, epochs = 500, eta = 0.01,
				 alpha = 0.0, decrease_const = 0.0, shuffle = True,
				 minibatch = 1, random_state = None):
		'''
		@brief Constructor
		@param n_output: # of output units, in this case 10;
		            	 Also, # of units in the output layer
		       n_features: m of the X matrix, in this case 784;
		                   Also, # of units in the input layer
		       n_hidden: # of units in the hidden layer
		       l1: Lambda for L1 Regulization
		       l2: Lambda for L2 Regulization to decrease the degree of 
		           overfitting
		       epochs: # of passes over the training set
		       eta: The Learning Rate
		       alpha: Paramter to multiply to previous Gradient in order to 
		              add to current delta(w_t) (Value of weight change)
					  t: epoch t
			   decrease_const: Adaptive Learning Rate requires eta to decrease 
			                   under eta / (1 + t * d)
			                   t: epoch t
			   shuffle: Whether to shuffle Training Dataset to avoid cycles
			   Minibatch: Splitting the dataset into k mini-batches for fast learning
		'''
		np.random.seed(random_state)
		self.n_output = n_output
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.w1, self.w2 = self._initialize_weights()
		self.l1 = l1
		self.l2 = l2
		self.epochs = epochs
		self.eta = eta
		self.alpha = alpha
		self.decrease_const = decrease_const
		self.shuffle = shuffle
		self.minibatch = minibatch

	def _initialize_weights(self):
		'''
		@brief Initilize the Weight Matrix for both Input->Hidden (self.w1_)
		       and Hidden->Output (self.w2_)
		@note self.w1_ shape is h * (m+1); self.w2_ shape is n * (h+1);
			  in which m is # of features;
					   h is # of hidden units;
					   n is # of output units
		'''
		w1 = np.random.uniform(-1.0, 1.0,
							   size = self.n_hidden * (self.n_features + 1))
		w1 = w1.reshape(self.n_hidden, self.n_features + 1)
		w2 = np.random.uniform(-1.0, 1.0,
							   size = self.n_output * (self.n_hidden + 1))
		w2 = w2.reshape(self.n_output, self.n_hidden + 1)
		return w1, w2


print("Implementation of MLP Neural Network Object")
print("And its Application to the Hand Written Digits")

print("1. Read Add Data from CSV Files")
X_train = np.genfromtxt("train_img.csv",
						dtype = int,
						delimiter = ",")
y_train = np.genfromtxt("train_labels.csv",
						dtype = int,
						delimiter = ",")
print("Training Data Size:")
print("Rows: {0}, Columns: {1}".format(X_train.shape[0], X_train.shape[1]))

X_test = np.genfromtxt("test_img.csv",
					   dtype = int,
					   delimiter = ",")
y_test = np.genfromtxt("test_labels.csv",
					   dtype = int,
					   delimiter = ",")
print("\nTesting Data Size:")
print("Rows: {0}, Columns: {1}".format(X_test.shape[0], X_test.shape[1]))

print("\n2. Generate the ANN Object and Train the Training Dataset")
nn = NeuralNetMLP(n_output = 10,
				  n_features = X_train.shape[1],
				  n_hidden = 50,
				  l1 = 0.0,
				  l2 = 0.1,
				  epochs = 1000,
				  eta = 0.001,
				  alpha = 0.001,
				  decrease_const = 0.00001,
				  shuffle = True,
				  minibatch = 50,
				  random_state = 1)