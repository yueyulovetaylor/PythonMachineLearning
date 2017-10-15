# Implement some user test cases for Clean and Preprocess procedure
import sys
sys.path.append('./')
import CleanPreProcess as cp
from nltk.corpus import stopwords

print('Implement some user test cases for Clean and Preprocess procedure')
str1 = 'is seven.<br /><br />Title (Brazil): Not Available'
str1AfterProcess = cp.preprocessor(str1)
print('Example 1: ' + str1)
print('After Processing: ' + str1AfterProcess)

str2 = '</a>This :) is :( a test :-)!'
str2AfterProcess = cp.preprocessor(str2)
print('Example 2: ' + str1)
print('After Processing: ' + str2AfterProcess)

str3 = 'runners like running and thus they run'
print('Example 3 Tokenize: ' + str3)
withStopWordsArr = cp.tokenizer_porter(str3)
print(withStopWordsArr)

print('Clean up stopwords and we get')
stop = stopwords.words('english')
noStopWordsArr = [w for w in withStopWordsArr[-10:] if w not in stop]
print(noStopWordsArr)