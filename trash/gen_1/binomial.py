from scipy import *
import scipy.io
import scipy.ndimage
import numpy as np
import scipy.optimize as spoptim
import numpy.random
import matplotlib
#matplotlib.use('Agg') # When running on cluster, plots cannot be shown and this must be used
import matplotlib.pyplot as plt
import time
import sys
plt.rc('image', cmap='viridis')
from scipy import optimize
numpy.random.seed(13)
import scipy.special as sp

N = 40
n = 9
days = 30
p = 1/days

comb = sp.comb(N,n)
print(comb)
logcomb = np.log(comb)
print(logcomb)

# log prob
logprob = N*(log(days-1) - log(days)) - n*log(days-1)
summmm = logcomb + logprob
print("Log probability:",summmm)
print("Probability:",np.exp(summmm))
