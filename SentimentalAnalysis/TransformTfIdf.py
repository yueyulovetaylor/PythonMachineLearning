# An Example of scripts that transform Three Sentences to Feature Vector
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

print('An Example of scripts that transform Three Sentences to Feature Vector')
print('and Calculate tf-idf Value')
print('1. Transform samples into Feature Vectors')
count = CountVectorizer()
docs = np.array([
	'The sun is shining',
	'The weather is sweet',
	'The sun is shining and the weather is sweet'])

print('Input sentences are')
for i in range(0, docs.shape[0]):
	print(docs[i])

bag = count.fit_transform(docs)
print('\nThe overall word count is')
print(count.vocabulary_)

print('\nThe feature vector is')
print(bag.toarray())

print('2. Calculate the tf-idf value')
tfidf = TfidfTransformer()
np.set_printoptions(precision = 2)
print('Tfidf transform Matrix is')
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())