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

N_seeds = 20
lambda_strength_array = [1.01,1.1,1.2,1.3,1.4,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5,6,7,8,9,10]
T_100 = np.load("mean_rmse_values-T-100-up-to-lambda-10.npy")
std_100_array = np.load("std_values-T-100-up-to-lambda-10.npy")

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, 4)]

plt.figure()
plt.title("Average RMSE with background noise 0.5")
plt.xlabel("Expected number of spikes in a bin")
plt.ylabel("RMSE")
#plt.plot(lambda_strength_array, T_10, ".", label="T=10", color = colors[0])
plt.errorbar(x=lambda_strength_array, y=T_100, yerr=(1.96*std_100_array/np.sqrt(N_seeds)), fmt="_", label="T=100", color = colors[1])
#plt.plot(lambda_strength_array, T_100, ".", label="T=100", color = colors[1])
#plt.plot(lambda_strength_array, T_100 - 1.96*std_100_array/np.sqrt(N_seeds), "_", label="T=100", color = colors[1])
#plt.plot(lambda_strength_array, T_100 + 1.96*std_100_array/np.sqrt(N_seeds), "_", label="T=100", color = colors[1])
#plt.plot(lambda_strength_array, T_1000, ".", label="T=1000", color = colors[2])
plt.legend(loc="upper right")
plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness.png")
plt.show()
