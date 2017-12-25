# Implementation of Activation Function
import numpy as np
import matplotlib.pyplot as plt

print('Implementation of Activation Function')
print('1. Logistic Sigmoid Function')
X = np.array([[1, 1.4, 1.5]])
w = np.array([0.0, 0.2, 0.4])

def net_input(X, w):
	z = X.dot(w)
	return z

def logistic(z):
	return 1.0 / (1.0 + np.exp(-z))

def logistic_activate(X, w):
	z = net_input(X, w)
	return logistic(z)

print('P(y = 1|X) = {0}'.format(round(logistic_activate(X, w)[0], 3)))

print('\nIf we want to deal with a Matrix Manipulation')
W = np.array([[1.1, 1.2, 1.3, 0.5],
			  [0.1, 0.2, 0.4, 0.1],
			  [0.2, 0.5, 2.1,1.9]])
A = np.array([[1.0], [0.1], [0.3], [0.7]])
Z = W.dot(A)
y_probas = logistic(Z)
print("Probabilities Vector:\n {0}".format(y_probas))

print('\n2. Softmax')
def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

def softmax_activation(X, w):
	z = net_input(X, w)
	return softmax(z)

y_probas = softmax(Z)
print('Probabilities Vector:\n {0}'.format(y_probas))
y_class = np.argmax(Z, axis = 0)
print('Predicted class label: {0}'.format(y_class[0]))

print('\n3. Hyperbolic Tangent')
def tanh(z):
	e_p = np.exp(z)
	e_m = np.exp(-z)
	return (e_p - e_m) / (e_p + e_m)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

plt.ylim([-1.5, 1.5])
plt.xlabel("Net input $z$")
plt.ylabel('Activation $\phi(z)$')
plt.axhline(1, color = 'black', linestyle = '--')
plt.axhline(0.5, color = 'black', linestyle = '--')
plt.axhline(0, color = 'black', linestyle = '--')
plt.axhline(-1, color = 'black', linestyle = '--')

plt.plot(z, tanh_act,
		 linewidth = 2,
		 color = 'black',
		 label = 'tanh')

plt.plot(z, log_act,
		 linewidth = 2,
		 color = 'lightgreen',
		 label = 'logistic')

plt.legend(loc = 'lower right')
plt.show()