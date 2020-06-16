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

#lambda_strength_array = [0.01,0.1,0.3,0.5,0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#lambda_strength_array = [0.01,0.1,0.3,0.5,0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#T_10 = np.load("mean_rmse_values-T-10-up-to-lambda-15.npy")
#T_100 = np.load("mean_rmse_values-T-100-up-to-lambda-15.npy")
##T_1000 = np.load("mean_rmse_values-T-1000-up-to-lambda-3.npy")
#T_1000 = np.load("mean_rmse_values-T-1000-up-to-lambda-3-seeds013511.npy")

#lambda_strength_array = [1.01,1.1,1.2,1.3,1.4,1.5,1.75,2,2.5,3,4,5,6]
#T_100 = np.load("mean_rmse_values-T-100-up-to-lambda-6.npy")
#T_1000 = np.load("mean_rmse_values-T-1000-up-to-lambda-6.npy")

N_T_lengths = 6
N_seeds = 20
lambda_strength_array = [1.01,1.1,1.2,1.3,1.4,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5,6,7,8,9,10]

T_10 = np.load("mean_rmse_values-T-10-up-to-lambda-10.npy")
std_10_array = np.load("std_values-T-10-up-to-lambda-10.npy")
T_100 = np.load("mean_rmse_values-T-100-up-to-lambda-10.npy")
std_100_array = np.load("std_values-T-100-up-to-lambda-10.npy")
T_1000 = np.load("mean_rmse_values-T-1000-up-to-lambda-10.npy")
std_1000_array = np.load("std_values-T-1000-up-to-lambda-10.npy")
T_2000 = np.load("mean_rmse_values-T-2000-up-to-lambda-10.npy")
std_2000_array = np.load("std_values-T-2000-up-to-lambda-10.npy")
T_3162 = np.load("mean_rmse_values-T-3162-up-to-lambda-1.3.npy")
std_3162_array = np.load("std_values-T-3162-up-to-lambda-1.3.npy")
T_5000 = np.load("mean_rmse_values-T-5000-up-to-lambda-2.npy")
std_5000_array = np.load("std_values-T-5000-up-to-lambda-2.npy")

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N_T_lengths)]

plt.figure()
plt.title("Average RMSE with background noise 1.0")
plt.xlabel("Expected number of spikes in a bin")
plt.ylabel("RMSE")
#### Errorbar
plt.errorbar(x=lambda_strength_array, y=T_10, yerr=(1.96*std_10_array/np.sqrt(N_seeds)), fmt="-", label="T=10", color = colors[0])
plt.errorbar(x=lambda_strength_array, y=T_100, yerr=(1.96*std_100_array/np.sqrt(N_seeds)), fmt="-", label="T=100", color = colors[1])
plt.errorbar(x=lambda_strength_array, y=T_1000, yerr=(1.96*std_1000_array/np.sqrt(N_seeds)), fmt="-", label="T=1000", color = colors[2])
plt.errorbar(x=lambda_strength_array, y=T_2000, yerr=(1.96*std_2000_array/np.sqrt(N_seeds)), fmt="-", label="T=2000", color = colors[3])
#plt.errorbar(x=lambda_strength_array, y=T_3162, yerr=(1.96*std_3162_array/np.sqrt(N_seeds)), fmt="_", label="T=3162", color = colors[4])
#plt.errorbar(x=lambda_strength_array, y=T_5000, yerr=(1.96*std_5000_array/np.sqrt(N_seeds)), fmt="_", label="T=5000", color = colors[5])
#### Just mean
#plt.plot(lambda_strength_array, T_10, "-", label="T=10", color = colors[0])
#plt.plot(lambda_strength_array, T_100, "-", label="T=100", color = colors[1])
#plt.plot(lambda_strength_array, T_1000, "-", label="T=1000", color = colors[2])
#plt.plot(lambda_strength_array, T_2000, "-", label="T=2000", color = colors[3])
##plt.plot(lambda_strength_array, T_3162, "-", label="T=3162", color = colors[4])
##plt.plot(lambda_strength_array, T_5000, "-", label="T=5000", color = colors[5])

plt.legend(loc="upper right")
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness.png")
plt.show()





#plt.plot(lambda_strength_array, T_100 - 1.96*std_100_array/np.sqrt(N_seeds), "_", label="T=100", color = colors[1])
#plt.plot(lambda_strength_array, T_100 + 1.96*std_100_array/np.sqrt(N_seeds), "_", label="T=100", color = colors[1])
