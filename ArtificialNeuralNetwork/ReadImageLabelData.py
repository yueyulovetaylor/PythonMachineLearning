# Implementation of Reading Image and Label Data into two Arrays and 
# store the corresponding data into a CSV file
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind = "train"):
	labels_path = os.path.join(path, 
							   "{0}-labels-idx1-ubyte".format(kind))
	images_path = os.path.join(path,
							   "{0}-images-idx3-ubyte".format(kind))
	
	# Read out Labels Array, Dimension of labels Array is n * 1
	# n is number of Samples
	labels = None
	with open(labels_path, "rb") as lbpath:
		magic, n = struct.unpack(">II",
								 lbpath.read(8))
		labels = np.fromfile(lbpath,
							 dtype = np.uint8)

	# Read out Images Matrix, Dimension of Image Matrix is n * m
	# m is number of Features
	images = None
	with open(images_path, "rb") as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII",
											   imgpath.read(16))
		images = np.fromfile(imgpath, 
							 dtype = np.uint8).reshape(len(labels), 784)
	
	return images, labels

print("Implementation of Reading Image And Label Data and Store in CSV File")
print("1. Read Image and Label Data from Raw Files")
currentDirec = "/Users/yyu196/Codes/PythonMachineLearning/ArtificialNeuralNetwork"

X_train, y_train = load_mnist(currentDirec, kind = "train")
print("Training Data Size:")
print("Rows: {0}, Columns: {1}".format(X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist(currentDirec, kind = "t10k")
print("\nTesting Data Size:")
print("Rows: {0}, Columns: {1}".format(X_test.shape[0], X_test.shape[1]))

print("\n2. Plot the first ten Sample Hand Written Numbers")
fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True)
ax = ax.flatten()
for i in range(10):
	img = X_train[y_train == i][0].reshape(28, 28)
	ax[i].imshow(img, cmap = "Greys", interpolation = "nearest")

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n3. Plot Different Handwrittings for the same Digit")
fig, ax = plt.subplots(nrows = 5,
					  ncols = 5, 
					  sharex = True, 
					  sharey = True)
ax = ax.flatten()
for i in range(25):
	img = X_train[y_train==7][i].reshape(28, 28)
	ax[i].imshow(img, cmap = "Greys", interpolation = "nearest")

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n4. Store the Data into a CSV File")
np.savetxt("train_img.csv", X_train,
		   fmt = "%i", delimiter = ",")
np.savetxt("train_labels.csv", y_train,
		   fmt = "%i", delimiter = ",")
np.savetxt("test_img.csv", X_test,
		   fmt = "%i", delimiter = ",")
np.savetxt("test_labels.csv", y_test,
		   fmt = "%i", delimiter = ",")