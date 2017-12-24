# Implementation of MLP Neural Network Object (Class)
# Application of ANN to Hand Written Digits' Data
import numpy as np
from scipy.special import expit
import sys
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Implementations of MLP Neural Network Object
class NeuralNetMLP(object):
	def __init__(self, n_output, n_features, n_hidden = 30,
				 l1 = 0.0, l2 = 0.0, epochs = 500, eta = 0.01,
				 alpha = 0.0, decrease_const = 0.0, shuffle = True,
				 minibatches = 1, random_state = None):
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
		self.minibatches = minibatches

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

	def _sigmoid(self, z):
		return expit(z) # Equivalent to 1.0 / (1.0 + np.exp(-z))

	def _sigmoid_gradient(self, z):
		'''
		@brief d(phi(z)/z) = phi(z) / (1 - phi(z))
			   See Notes for detailed Mathematical Proof
		'''
		sg = self._sigmoid(z)
		return sg * (1.0 - sg)

	def _encode_labels(self, y, k):
		'''
		@brief Encoded Labels should be a n * n_samples Matrix
		       for which can be compared to output units Matrix (a3)
		'''
		onehot = np.zeros((k, y.shape[0]))
		for idx, val in enumerate(y):
			onehot[val, idx] = 1.0
		return onehot

	def _add_bias_unit(self, X, how = "Column"):
		'''
		@brief Append a Bias Unit to the end of the assigned demension
		       which could be "Column" or "Row"
		'''
		if how == 'column':
			X_new = np.ones((X.shape[0], X.shape[1] + 1))
			X_new[:, 1:] = X
		elif how == 'row':
			X_new = np.ones((X.shape[0] + 1, X.shape[1]))
			X_new[1:, :] = X
		else:
			raise AttributeError('`how` must be `column` or `row`')
		return X_new

	def _feedforward(self, X, w1, w2):
		'''
		@brief Feed Forward using Neural Network Algorithms
			   return input data (a1); hidden data (z2, a2) and output data (z3, a3)
		'''
		a1 = self._add_bias_unit(X, how = "column")
		z2 = w1.dot(a1.T)
			# w1: h * (m + 1); a1: n_sample * (m + 1)
			# So, z2: h * n_samples
		a2 = self._sigmoid(z2)
		a2 = self._add_bias_unit(a2, how = "row")
			# z2: h * n_samples; So a2: (h + 1) * n_samples
		z3 = w2.dot(a2)
			# w2: n * (h + 1); a2: (h + 1) * n_samples
			# So z3: n * n_samples
		a3 = self._sigmoid(z3)
		return a1, z2, a2, z3, a3

	def _L1_reg(self, lambda_, w1, w2):
		'''
		@brief Get the L1 Regularization Term Value
		'''
		return (lambda_ / 2.0) * \
			   (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

	def _L2_reg(self, lambda_, w1, w2):
		'''
		@brief Get the L2 Regulazrization Term Value
		'''
		return (lambda_ / 2.0) * \
			   (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

	def _get_cost(self, y_enc, output, w1, w2):
		'''
		@brief Get the each output unit cost based on Logistic Regression Cost Function
		       Sum up all units and samples cost (n * n_samples Data Point) 
		'''
		term1 = -y_enc * (np.log(output))
		term2 = (1.0 - y_enc) * np.log(1.0 - output)
		cost = np.sum(term1 - term2)
		L1_term = self._L1_reg(self.l1, w1, w2)
		L2_term = self._L2_reg(self.l2, w1, w2)
		cost = cost + L1_term + L2_term
		return cost

	def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
		'''
		@brief Get the Gradient for w1 and w2 
		'''
		# Get the Backpropagation Gradient
		sigma3 = a3 - y_enc
		z2 = self._add_bias_unit(z2, how='row')
		sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
		sigma2 = sigma2[1:, :]
		grad1 = sigma2.dot(a1)
		grad2 = sigma3.dot(a2.T)

		# regularize
		grad1[:, 1:] += self.l2 * w1[:, 1:]
		grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
		grad2[:, 1:] += self.l2 * w2[:, 1:]
		grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

		return grad1, grad2

	def predict(self, X):
		'''
		@brief Use Feed Forward to calculate the Most Possible y prediction value
		'''
		a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
		y_pred = np.argmax(z3, axis = 0)
		return y_pred

	def fit(self, X, y, print_progress = False):
		'''
		@brief Training the Dataset to get all Parameters, aka w1 and w1
		       so that we can later use to predict the Test Dataset
		'''
		self.cost_ = []
		X_data, y_data = X.copy(), y.copy()
		y_enc = self._encode_labels(y, self.n_output)

		delta_w1_prev = np.zeros(self.w1.shape)
		delta_w2_prev = np.zeros(self.w2.shape)

		for i in range(self.epochs):
			# adaptive learning rate
			self.eta /= (1 + self.decrease_const*i)

			if print_progress:
				sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
				sys.stderr.flush()

			if self.shuffle:
				idx = np.random.permutation(y_data.shape[0])
				X_data, y_enc = X_data[idx], y_enc[:, idx]

			mini = np.array_split(range(y_data.shape[0]), self.minibatches)
			for idx in mini:
				# feedforward
				a1, z2, a2, z3, a3 = self._feedforward(X_data[idx],
													   self.w1,
													   self.w2)
				cost = self._get_cost(y_enc=y_enc[:, idx],
									  output=a3,
									  w1=self.w1,
									  w2=self.w2)
				self.cost_.append(cost)

				# compute gradient via backpropagation
				grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
												  a3=a3, z2=z2,
												  y_enc=y_enc[:, idx],
												  w1=self.w1,
												  w2=self.w2)

				delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
				self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
				self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
				delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

		return self

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
				  minibatches = 50,
				  random_state = 1)
nn.fit(X_train, y_train, print_progress = True)

print("\n\n3. Plot Cost with Epochs")
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel("Cost")
plt.xlabel("Epochs * 50")
plt.tight_layout()
plt.show()

print("\n4. Plot in Batches")
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color = 'red')
plt.ylim([0, 2000])
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.tight_layout()
plt.show()
