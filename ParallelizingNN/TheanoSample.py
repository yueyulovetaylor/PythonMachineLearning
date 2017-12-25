# Implementation of Simple Samples of Theano
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

print('Simple Samples of Theano')

theano.config.floatX = 'float32'
print("Configure float type as {0}".format(theano.config.floatX))

print('1. A simple calculation sample')
# initialize 
x1 = T.scalar()
w1 = T.scalar()
w0 = T.scalar()
z1 = w1 * x1 + w0

# compile
net_input = theano.function(inputs = [w1, x1, w0],
							outputs = z1)

# execute
print('Net input: {0}'.format(round(net_input(2.0, 1.0, 0.5), 2)))

print('\n2. With Vector (Array)')
# initialize
x = T.fmatrix(name = 'x')
x_sum = T.sum(x, axis = 0)

# compile 
calc_sum = theano.function(inputs = [x], outputs = x_sum)

#execute
ary = np.array([[1, 2, 3], [1, 2, 3]],
				dtype = theano.config.floatX)
print("Vector Sum: {0}".format(calc_sum(ary)))

print("\n3. Usage of Shared-Variables, Updates and Givens")
#initialize
data = np.array([[1, 2, 3]], dtype = theano.config.floatX)
x = T.fmatrix(name = 'x')
w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype = theano.config.floatX))
z = x.dot(w.T)
update = [[w, w + 1]]

# compile 
net_input = theano.function(inputs = [],
							updates = update,
							givens = {x: data},
							outputs = z)

# execute
for i in range(5):
	print("Iteration {0} -- Value {1}".format(i, net_input()))

print("\n4. Theano on Linear Regression")
X_train = np.asarray([[0.0], [1.0],
					  [2.0], [3.0],
					  [4.0], [5.0],
					  [6.0], [7.0],
					  [8.0], [9.0]],
					  dtype = theano.config.floatX)
y_train = np.asarray([1.0, 1.3,
					  3.1, 2.0,
					  5.0, 6.3,
					  6.6, 7.4,
					  8.0, 9.0],
					  dtype = theano.config.floatX)

def train_linreg(X_train, y_train, eta, epochs):
	'''
	@brief Use Theano to train Linear Regression
	'''
	theano.config.floatX = 'float32'
	costs = []
	#Initialization Array
	eta0 = T.fscalar('eta0')
	y = T.fvector(name = 'y')
	X = T.fmatrix(name = 'X')
	w = theano.shared(np.zeros(shape = (X_train.shape[1] + 1),
							   dtype = theano.config.floatX),
					  name = 'w')

	#Calculate cost
	net_input = T.dot(X, w[1:]) + w[0]
	errors = y - net_input
	cost = T.sum(T.pow(errors, 2))

	#Perform Gradient Update
	gradient = T.grad(cost, wrt = w)
	update = [(w, w - eta0 * gradient)]

	#Compile
	train = theano.function(inputs = [eta0],
							outputs = cost,
							updates = update,
							givens = {X: X_train,
									  y: y_train})

	for _ in range(epochs):
		costs.append(train(eta))

	return costs, w

print('Plot Epoch vs. Cost')
costs, w = train_linreg(X_train, y_train, eta = 0.001, epochs = 10)
plt.plot(range(1, len(costs) + 1), costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

def predict_linreg(X, w):
	Xt = T.matrix(name = 'X')
	net_input = T.dot(Xt, w[1:]) + w[0]
	predict = theano.function(inputs = [Xt],
							  givens = {w: w},
							  outputs = net_input)
	return predict(X)

print('Plot y vs X (Scatter and Prediction Line)')
plt.scatter(X_train, y_train, marker = 's', s = 50)
plt.plot(range(X_train.shape[0]),
		 predict_linreg(X_train, w),
		 color = 'grey',
		 marker = 'o',
		 markersize = 4,
		 linewidth = 3)
plt.xlabel('X')
plt.ylabel('y')
plt.show()
