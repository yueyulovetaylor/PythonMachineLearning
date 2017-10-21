# Use Pickle to Serialize the IMDB Data Training Model into a pkl file
import numpy as np
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

import pyprind
import pickle
import os

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

print('Use Pickle to Serialize the IMDB Data Training Model into a pkl file')
print('1. Reuse the code from Train IMDB Data to get the classification object')
vect = HashingVectorizer(decode_error = 'ignore',
						 n_features = 2 ** 21,
						 preprocessor = None,
						 tokenizer = tokenizer)
clf = SGDClassifier(loss = 'log', random_state = 1, n_iter = 1)
doc_stream = stream_docs(path = '../SentimentalAnalysis/movie_data.csv')

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
	X_train, y_train = get_minibatch(doc_stream, size = 1000)
	if not X_train:
		break
	X_train = vect.transform(X_train)
	clf.partial_fit(X_train, y_train, classes = classes)
	pbar.update()

X_test, y_test = get_minibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print('Accuracy: {0}'.format(round(clf.score(X_test, y_test), 3)))

clf = clf.partial_fit(X_test, y_test)

print('2. Serialize the Model into a pkl file')
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
	os.makedirs(dest)
pickle.dump(stop,
			open(os.path.join(dest, 'stopwords.pkl'), 'wb'))
pickle.dump(clf,
			open(os.path.join(dest, 'classifier.pkl'), 'wb'))