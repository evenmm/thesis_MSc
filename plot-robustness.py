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

# scp evenmm@idun-login1.hpc.ntnu.no:/home/evenmm/Master/thesis/m_s_arrays/. m_s_arrays/.

N_seeds = 20 # that we average over
baseline_lambda_value = 0.5
T_array = [200,500,1000,2000,3000,5000]
tuning_difference_array = [0.01,0.1,0.2,0.3,0.4,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,7,8,9]
N_T_values = len(T_array)
N_lambdas = len(tuning_difference_array)
mean_rmse_values = np.zeros((N_T_values,N_lambdas))
sosq_values = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
for T_index in range(N_T_values):
    for lambda_index in range(N_lambdas):
        mean_name = "m_s_arrays/m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(lambda_index) + ".npy"
        sosq_name = "m_s_arrays/s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(lambda_index) + ".npy"
        mean_rmse_values[T_index, lambda_index] = np.load(mean_name)
        sosq_values[T_index, lambda_index] = np.load(sosq_name)

peak_lambda_array = [baseline_lambda_value + tuning_difference_array[i] for i in range(len(tuning_difference_array))]
colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N_T_values)]

plt.figure()
plt.title("Average RMSE with background noise 1.0")
plt.xlabel("Expected number of spikes in a bin at peak of tuning")
plt.ylabel("RMSE")
for T_index in range(N_T_values):
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values[T_index], yerr=(2.093*(np.sqrt(sosq_values[T_index]/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T="+str(T_array[T_index]), color = colors[T_index])
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.vlines(1, 0, 3.8, label="Background tuning level")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness.png")
plt.show()


"""
# Mean values 
#T_10 = np.load("mean_rmse_values-base-lambda-0.5T-10-up-to-lambda-9.5.npy")
#T_100 = np.load("mean_rmse_values-base-lambda-0.5T-100-up-to-lambda-9.5.npy")
T_200 = np.load("mean_rmse_values-base-lambda-0.5T-200-up-to-lambda-9.5.npy")
#T_500 = np.load("mean_rmse_values-base-lambda-0.5T-500-up-to-lambda-9.5.npy")
#T_1000 = np.load("mean_rmse_values-base-lambda-0.5T-1000-up-to-lambda-9.5.npy")
#T_2000 = np.load("mean_rmse_values-base-lambda-0.5T-2000-up-to-lambda-9.5.npy")
#T_3162 = np.load("mean_rmse_values-base-lambda-0.5T-3000-up-to-lambda-9.5.npy")
#T_5000 = np.load("mean_rmse_values-base-lambda-0.5T-5000-up-to-lambda-9.5.npy")

# Sum of squared errors
soSqDev_200_array = np.load("sum_of_squared_deviation_values-base-lambda-0.5T-200-up-to-lambda-9.5.npy")

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, 6)]
# 95 % confidence intervals with (20-1) degrees of freedom
# t_alpha/2 = 2.093
# S = np.sqrt(SoSqDev/(N_seeds - 1))

plt.figure()
plt.title("Average RMSE with background noise 1.0")
plt.xlabel("Expected number of spikes in a bin at peak of tuning")
plt.ylabel("RMSE")
#### Errorbar
#plt.errorbar(x=peak_lambda_array, y=T_10, yerr=(2.093*(np.sqrt(soSqDev_10_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=10", color = colors[0])
#plt.errorbar(x=peak_lambda_array, y=T_100, yerr=(2.093*(np.sqrt(soSqDev_100_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=100", color = colors[1])
plt.errorbar(x=peak_lambda_array, y=T_200, yerr=(2.093*(np.sqrt(soSqDev_200_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=200", color = colors[0])
#plt.errorbar(x=peak_lambda_array, y=T_500, yerr=(2.093*(np.sqrt(soSqDev_500_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=500", color = colors[1])
#plt.errorbar(x=peak_lambda_array, y=T_1000, yerr=(2.093*(np.sqrt(soSqDev_1000_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=1000", color = colors[2])
#plt.errorbar(x=peak_lambda_array, y=T_2000, yerr=(2.093*(np.sqrt(soSqDev_2000_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=2000", color = colors[3])
#plt.errorbar(x=peak_lambda_array, y=T_3162, yerr=(2.093*(np.sqrt(soSqDev_3162_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=3162", color = colors[4])
#plt.errorbar(x=peak_lambda_array, y=T_5000, yerr=(2.093*(np.sqrt(soSqDev_5000_array/(N_seeds-1)))/np.sqrt(N_seeds)), fmt="-", label="T=5000", color = colors[5])
#### Just mean
#plt.plot(peak_lambda_array, T_10, "-", label="T=10", color = colors[0])
#plt.plot(peak_lambda_array, T_100, "-", label="T=100", color = colors[1])
#plt.plot(peak_lambda_array, T_1000, "-", label="T=1000", color = colors[2])
#plt.plot(peak_lambda_array, T_2000, "-", label="T=2000", color = colors[3])
##plt.plot(peak_lambda_array, T_3162, "-", label="T=3162", color = colors[4])
##plt.plot(peak_lambda_array, T_5000, "-", label="T=5000", color = colors[5])
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.vlines(1, 0, 3.8, label="Background tuning level")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness.png")
plt.show()





#plt.plot(peak_lambda_array, T_100 - 2.093*(np.sqrt(soSqDev_100_array/(N_seeds-1)))/np.sqrt(N_seeds), "_", label="T=100", color = colors[1])
#plt.plot(peak_lambda_array, T_100 + 2.093*(np.sqrt(soSqDev_100_array/(N_seeds-1)))/np.sqrt(N_seeds), "_", label="T=100", color = colors[1])

#T_10 = np.load("mean_rmse_values-T-10-up-to-lambda-10.npy")
#T_100 = np.load("mean_rmse_values-T-100-up-to-lambda-10.npy")
#T_200 = np.load("mean_rmse_values-T-200-up-to-lambda-10.npy")
#T_500 = np.load("mean_rmse_values-T-500-up-to-lambda-10.npy")
#T_1000 = np.load("mean_rmse_values-T-1000-up-to-lambda-10.npy")
#T_2000 = np.load("mean_rmse_values-T-2000-up-to-lambda-5.npy")
#T_3162 = np.load("mean_rmse_values-T-3162-up-to-lambda-10.npy")
#T_5000 = np.load("mean_rmse_values-T-5000-up-to-lambda-10.npy")
# np.std (Old)
#soSqDev_10_array = np.load("std_values-T-10-up-to-lambda-10.npy")
#soSqDev_100_array = np.load("std_values-T-100-up-to-lambda-10.npy")
#soSqDev_1000_array = np.load("std_values-T-1000-up-to-lambda-10.npy")
#soSqDev_2000_array = np.load("std_values-T-2000-up-to-lambda-10.npy")
#soSqDev_3162_array = np.load("std_values-T-3162-up-to-lambda-10.npy")
#soSqDev_5000_array = np.load("std_values-T-5000-up-to-lambda-10.npy")
#soSqDev_10_array = np.load("sum_of_squared_deviation_values-T-10-up-to-lambda-10.npy")
#soSqDev_100_array = np.load("sum_of_squared_deviation_values-T-100-up-to-lambda-10.npy")
#soSqDev_200_array = np.load("sum_of_squared_deviation_values-T-200-up-to-lambda-10.npy")
#soSqDev_500_array = np.load("sum_of_squared_deviation_values-T-500-up-to-lambda-10.npy")
#soSqDev_1000_array = np.load("sum_of_squared_deviation_values-T-1000-up-to-lambda-10.npy")
#soSqDev_2000_array = np.load("sum_of_squared_deviation_values-T-2000-up-to-lambda-5.npy")
#soSqDev_3162_array = np.load("sum_of_squared_deviation_values-T-3162-up-to-lambda-1.3.npy")
#soSqDev_5000_array = np.load("sum_of_squared_deviation_values-T-5000-up-to-lambda-2.npy")
"""
