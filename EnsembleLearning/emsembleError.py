# Implement the Emsemble Error vs. Base Error
from scipy.misc import comb
import math
import numpy as np
import matplotlib.pyplot as plt

def ensemble_error(n_classifier, error):
	# @description Calculate the Ensemble Error based on 
	#              the number of classifiers
	k_start = int(math.ceil(n_classifier / 2.0))
	    # The start of k should be the first number after (n/2)
	probs = [comb(n_classifier, k) *
	         (error ** k) * 
	         ((1 - error) ** (n_classifier - k))
	         for k in range(k_start, n_classifier + 1)]
	return sum(probs)

print('Ensemble Error')
print('1. Sample Calculation of Ensemble Error')
sampleEnsembleError = ensemble_error(11, 0.25)
print('When n_classifier = 11, error = 0.25, Ensemble Error is {0}'.format(
	round(sampleEnsembleError, 5)))

print('\n2. Calculate and plot Ensemble Error when Base Error changes from 0.0 to 1.0')
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier = 11, error = error)
              for error in error_range]
plt.plot(error_range, ens_errors, label = 'Ensemble Errors', linewidth = 2)
plt.plot(error_range, error_range, linestyle = '--', label = 'Base Error', 
	     linewidth = 2)
plt.xlabel('Base Error')
plt.ylabel('Base/Ensembe Error')
plt.legend(loc = 'upper left')
plt.grid()
plt.show()