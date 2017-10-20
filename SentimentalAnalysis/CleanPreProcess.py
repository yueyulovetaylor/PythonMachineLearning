# Implementation of Clean Data and Preprocessing Utilities
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def preprocessor(text):
	text = re.sub('<[^>]*>', '', text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
	return text

def tokenizer(text):
	return text.split()

def tokenizer_porter(text):
	porter = PorterStemmer()
	stop = stopwords.words('english')
	return [porter.stem(word) for word in text.split()]
