# Implement Training the Imdb Data
import pyprind
import pandas as pd
import os
import sys
sys.path.append('./')
import CleanPreProcess as cp
from nltk.corpus import stopwords

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

pbar = pyprind.ProgBar(50000)  
	# ProgBar: simple progression bar for shell scripts
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()

print('Implement Training the Imdb Data')
print('1. Copy data into a DataFrame object')
for s in {'test', 'train'}:
	for l in {'pos', 'neg'}:
		path = './aclImdb/{0}/{1}'.format(s, l)
		for file in os.listdir(path):
			with open(os.path.join(path, file)) as infile:
				txt = infile.read()
				df = df.append([[txt, labels[l]]], ignore_index = True)
				pbar.update()

df.columns = ['review', 'sentiment']
df['review'] = df['review'].apply(cp.preprocessor)

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

print('2. Transform the data to Use Logistic Regression to train the Imdb Data')
tfidf = TfidfVectorizer(strip_accents = None,
						lowercase = False,
						preprocessor = None)
stop = stopwords.words('english')
param_grid = [{'vect__ngram_range': [(1, 1)],
			   'vect__stop_words': [stop, None],
			   'vect__tokenizer': [cp.tokenizer,
			   					   cp.tokenizer_porter],
			   'clf__penalty': ['l1', 'l2'],
			   'clf__C': [1.0, 10.0, 100.0]},
			  {'vect__ngram_range': [(1, 1)],
			   'vect__stop_words': [stop, None],
			   'vect__tokenizer': [cp.tokenizer,
			   					   cp.tokenizer_porter],
			   'vect__use_idf': [False],
			   'vect__norm': [None],
			   'clf__penalty': ['l1', 'l2'],
			   'clf__C': [1.0, 10.0, 100.0]}
			 ]
lr_tfidf = Pipeline([('vect', tfidf),
					 ('clf', LogisticRegression(random_state = 0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
						   scoring = 'accuracy',
						   cv = 5,
						   verbose = 1,
						   n_jobs = -1)
gs_lr_tfidf.fit(X_train, y_train)
print('Best Parameters are')
print(gs_lr_tfidf.best_params_)

print('\nPrint final Training result:')
print('CV Accuracy {0}'.format(round(gs_lr_tfidf.best_score_, 3)))
clf = gs_lr_tfidf.best_estimator_
testScore = clf.score(X_test, y_test)
print('\Testing result {0}'.format(round(testScore, 3)))