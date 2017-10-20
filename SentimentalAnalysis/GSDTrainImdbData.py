# Use Stochastic Gradient Descent Approach to optimize the Classification Model
import numpy as np
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

import pyprind

stop = stopwords.words('english')

def tokenizer(text):
	text = re.sub('<[^>]*>', '', text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	tokenized = [w for w in text.split() if w not in stop]
	return tokenized

def stream_docs(path):
	with open(path, 'r') as csv:
		next(csv)    # Skip header
		for line in csv:
			text, label = line[:-3], int(line[-2])
			yield text, label

def get_minibatch(doc_stream, size):
	docs, y = [], []
	try:
		for _ in range(size):
			text, label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except StopIteration:
		return None, None
	return docs, y 

print('Use Stochastic Gradient Descent Approach to optimize the Classification Model')
print('1. Reading out the csv file; the first line is:')
print(next(stream_docs(path = './movie_data.csv')))

print('2. Train the Data')
print('Create a HashingVectorizer Object')
vect = HashingVectorizer(decode_error = 'ignore',
						 n_features = 2 ** 21,
						 preprocessor = None,
						 tokenizer = tokenizer)

print('Create the SGDClassifier Object')
clf = SGDClassifier(loss = 'log', random_state = 1, n_iter = 1)

print('Create StreamDocs Object')
doc_stream = stream_docs(path = './movie_data.csv')

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

print('Get the mini batches to train the whole dataset')
for _ in range(45):
	X_train, y_train = get_minibatch(doc_stream, size = 1000)
	if not X_train:
		break
	X_train = vect.transform(X_train)
	clf.partial_fit(X_train, y_train, classes = classes)
	pbar.update()

print('Use the clf model to get the test score')
X_test, y_test = get_minibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print('Accuracy: {0}'.format(round(clf.score(X_test, y_test), 3)))