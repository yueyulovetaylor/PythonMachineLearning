# Test Suite to Check the Performance of Pickle object
import pickle
import re
import os
from vectorizer import vect

import numpy as np

print('Test Suite to Check the Performance of Pickle object')
print('1. Load Pickle object')
clf = pickle.load(open(
				  os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

print('2. Predict the test sentence')
label = {0: 'negative', 1: 'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Example sentence: {0}'.format(example[0]))
print('Prediction: {0}'.format(label[clf.predict(X)[0]]))
print('Probability: {0}%'.format(round(np.max(clf.predict_proba(X)) * 100, 3)))