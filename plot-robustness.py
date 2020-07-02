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

# scp -r evenmm@idun-login1.hpc.ntnu.no:/home/evenmm/Master/thesis/m_s_arrays/ .

N_seeds = 20 # that we average over
degrees_of_freedom = N_seeds - 1 # in t distribution
print("Averaging over", N_seeds, "seeds.")
baseline_lambda_value = 0.5
T_array = [200,500,1000,2000,3000,5000] #[200,500,1000]
N_T_values = len(T_array)
tuning_difference_array = [0.01,0.1,0.2,0.3,0.4,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,7,8,9] #[0.01,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8]
finished_lambda_indexes = range(21) #[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #range(21) on a good day
N_lambdas = len(finished_lambda_indexes)
mean_rmse_values_3 = np.zeros((N_T_values,N_lambdas)) # mean rmse value with smoothingwindow 3 for PCA
sosq_values_3 = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
mean_rmse_values_5 = np.zeros((N_T_values,N_lambdas)) # mean rmse value with smoothingwindow 5 for PCA
sosq_values_5 = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
mean_rmse_values_10 = np.zeros((N_T_values,N_lambdas)) # mean rmse value with smoothingwindow 10 for PCA
sosq_values_10 = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
mean_rmse_values_L = np.zeros((N_T_values,N_lambdas)) # mean rmse with X estimate chosen based on L value out of three different smoothingwindow initializations
sosq_values_L = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
mean_rmse_values_RMSE = np.zeros((N_T_values,N_lambdas)) # mean rmse with X estimate chosen based on RMSE score out of three different smoothingwindow initializations
sosq_values_RMSE = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
for T_index in range(N_T_values): 
    for i in range(len(finished_lambda_indexes)): # i iterates in an array of lambda indices to get lambda values
        mean_name = "./m_s_arrays/3-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/3-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_3[T_index, i] = np.load(mean_name)
        sosq_values_3[T_index, i] = np.load(sosq_name)
        mean_name = "./m_s_arrays/5-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/5-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_5[T_index, i] = np.load(mean_name)
        sosq_values_5[T_index, i] = np.load(sosq_name)
        mean_name = "./m_s_arrays/10-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/10-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_10[T_index, i] = np.load(mean_name)
        sosq_values_10[T_index, i] = np.load(sosq_name)
        mean_name = "./m_s_arrays/L-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/L-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_L[T_index, i] = np.load(mean_name)
        sosq_values_L[T_index, i] = np.load(sosq_name)
        mean_name = "./m_s_arrays/RMSE-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/RMSE-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_RMSE[T_index, i] = np.load(mean_name)
        sosq_values_RMSE[T_index, i] = np.load(sosq_name)

peak_lambda_array = [baseline_lambda_value + tuning_difference_array[i] for i in finished_lambda_indexes]
colors = [plt.cm.viridis(t) for t in np.linspace(0, 0.8, N_T_values)]

# Just L value
plt.figure()
plt.title("Average RMSE with background noise " + str(baseline_lambda_value))
plt.xlabel("Expected number of spikes in a bin at peak of tuning")
plt.ylabel("RMSE")
for T_index in range(N_T_values):
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_L[T_index], yerr=(2.093*(np.sqrt(sosq_values_L[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt=".-", label="L choice.        T="+str(T_array[T_index]), color = plt.cm.viridis(0.8*T_index/N_T_values))
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.vlines(baseline_lambda_value, 0, 3.8, label="Background tuning level")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness-L-base-" + str(baseline_lambda_value) + ".png")

# RMSE
plt.figure()
plt.title("Average RMSE with background noise " + str(baseline_lambda_value))
plt.xlabel("Expected number of spikes in a bin at peak of tuning")
plt.ylabel("RMSE")
for T_index in range(N_T_values):
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_RMSE[T_index], yerr=(2.093*(np.sqrt(sosq_values_RMSE[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt=".-", label="RMSE choice. T="+str(T_array[T_index]), color = plt.cm.viridis(0.8*T_index/N_T_values))
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.vlines(baseline_lambda_value, 0, 3.8, label="Background tuning level")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness-RMSE-base-" + str(baseline_lambda_value) + ".png")

# Smoothingwidth = 5
plt.figure()
plt.title("Average RMSE with background noise " + str(baseline_lambda_value))
plt.xlabel("Expected number of spikes in a bin at peak of tuning")
plt.ylabel("RMSE")
for T_index in range(N_T_values):
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_5[T_index], yerr=(2.093*(np.sqrt(sosq_values_5[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt=".-", label="Width 5.         T="+str(T_array[T_index]), color = plt.cm.viridis(0.8*T_index/N_T_values))
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.vlines(baseline_lambda_value, 0, 3.8, label="Background tuning level")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness-5-base-" + str(baseline_lambda_value) + ".png")
plt.show()

# All together
plt.figure()
plt.title("Average RMSE with background noise " + str(baseline_lambda_value))
plt.xlabel("Expected number of spikes in a bin at peak of tuning")
plt.ylabel("RMSE")
for T_index in range(N_T_values):
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_RMSE[T_index], yerr=(2.093*(np.sqrt(sosq_values_RMSE[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt=".-", label="RMSE choice. T="+str(T_array[T_index]), color = plt.cm.viridis(0.1)) #plt.cm.viridis(T_index/N_T_values*0.8)
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_3[T_index], yerr=(2.093*(np.sqrt(sosq_values_3[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt="--", label="Width 3.         T="+str(T_array[T_index]), color = plt.cm.viridis(0.25))
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_5[T_index], yerr=(2.093*(np.sqrt(sosq_values_5[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt="--", label="Width 5.         T="+str(T_array[T_index]), color = plt.cm.viridis(0.5))
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_10[T_index], yerr=(2.093*(np.sqrt(sosq_values_10[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt="--", label="Width 10.       T="+str(T_array[T_index]), color = plt.cm.viridis(0.75))
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_L[T_index], yerr=(2.093*(np.sqrt(sosq_values_L[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt=".-", label="L choice.        T="+str(T_array[T_index]), color = plt.cm.viridis(0.9))
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.vlines(baseline_lambda_value, 0, 3.8, label="Background tuning level")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness-base-" + str(baseline_lambda_value) + ".png")
plt.show()
