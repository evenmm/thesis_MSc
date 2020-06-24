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
T_array = [200] #[200,500,1000,2000,3000,5000]
N_T_values = len(T_array)
tuning_difference_array = [0.01,0.1,0.2,0.3,0.4,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,7,8,9] #[0.01,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8]
finished_lambda_indexes = range(21) #[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #range(21) on a good day
N_lambdas = len(finished_lambda_indexes)
mean_rmse_values_FIVE = np.zeros((N_T_values,N_lambdas)) # mean rmse value with smoothingwindow 5 for PCA
sosq_values_FIVE = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
mean_rmse_values_L = np.zeros((N_T_values,N_lambdas)) # mean rmse with X estimate chosen based on L value out of three different smoothingwindow initializations
sosq_values_L = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
mean_rmse_values_RMSE = np.zeros((N_T_values,N_lambdas)) # mean rmse with X estimate chosen based on RMSE score out of three different smoothingwindow initializations
sosq_values_RMSE = np.zeros((N_T_values,N_lambdas)) # sum of squared errors
for T_index in range(N_T_values): 
    for i in range(len(finished_lambda_indexes)): # i iterates in an array of lambda indices to get lambda values
        mean_name = "./m_s_arrays/FIVE-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/FIVE-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_FIVE[T_index, i] = np.load(mean_name)
        sosq_values_FIVE[T_index, i] = np.load(sosq_name)
        mean_name = "./m_s_arrays/L-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/L-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_L[T_index, i] = np.load(mean_name)
        sosq_values_L[T_index, i] = np.load(sosq_name)
        mean_name = "./m_s_arrays/RMSE-m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        sosq_name = "./m_s_arrays/RMSE-s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[T_index]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
        mean_rmse_values_RMSE[T_index, i] = np.load(mean_name)
        sosq_values_RMSE[T_index, i] = np.load(sosq_name)

#for i in range(len(finished_lambda_indexes)): # i is a simple iterator in an array of lambda indices
#    mean_name = "./m_s_arrays/m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[1]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
#    sosq_name = "./m_s_arrays/s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[1]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
#    mean_rmse_values_FIVE[1, i] = np.load(mean_name)
#    sosq_values_FIVE[1, i] = np.load(sosq_name)
#    # 200 with RMSE instead of L:
#    mean_name = "./23-06-m_s_arrays-with-rmse-for-L/m-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[1]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
#    sosq_name = "./23-06-m_s_arrays-with-rmse-for-L/s-base-" + str(baseline_lambda_value) + "-T-" + str(T_array[1]) + "-lambda-index-" + str(finished_lambda_indexes[i]) + ".npy"
#    mean_rmse_values_FIVE[1, i] = np.load(mean_name)
#    sosq_values_FIVE[1, i] = np.load(sosq_name)

# Index 1 (and 0,1,2 for T=200) here are not correct!!
# T=200
#mean_rmse_values_FIVE[0] = [3.386329922899935, 3.386329922899935, 3.386329922899935, 3.0881359433460474, 2.717151603745104, 2.5680998540626865, 2.094590266666261, 1.759683160316344, 1.5491097871911512, 1.4799406340663215, 1.124234250015717, 0.8256776401545078, 0.6236000088691409, 0.5852147040875335, 0.4820780497753613, 0.5145531782243734, 0.43536713361959506, 0.6967643595836808, 0.7533607332009551, 0.6574862325557291, 0.8907439132902548]
#sosq_values_FIVE[0] = [1.3192931397419603, 1.3192931397419603, 1.3192931397419603, 6.488278438199352, 2.2327484387097924, 4.5753304709539755, 5.847672840541509, 4.319969238388791, 3.779659694044674, 5.352256958791524, 3.234261482187061, 1.8676439553748545, 0.4380667032311909, 0.4751254456215922, 0.3216960147075689, 1.2808226705854544, 0.6248465560475233, 5.2864811990065865, 7.439626491393257, 5.078195707327038, 7.799069279997841]
# T=500
#mean_rmse_values_FIVE[1] = [3.1533318960999273, 3.1518650352671247, 2.6178999236796257, 2.320482048678608, 2.068685983887247, 1.2852522292869926, 0.8994032216808627, 0.746546579912313, 1.1729691308972325, 1.2578503608595315, 1.288263110054407, 1.2578148219878593, 1.2873759508823068, 1.260347191270065, 1.1766776895488058, 1.2087756727325463, 1.15493381497601, 1.1420185500847475, 1.141135331523902, 1.1125260556771128, 1.1392651551592494]
#sosq_values_FIVE[1] = [0.7560500440118662, 0.15349066879889056, 0.07801970132136632, 0.05334935802899949, 0.11430684438529927, 0.8952474247685094, 0.11032693575986834, 0.9616738036477659, 0.6787680285322231, 0.2445739662225876, 0.2957435480200539, 0.09990701337324481, 0.22198812237200288, 0.08855621604057598, 0.14770294819931465, 0.17372303424443533, 0.2549674556238165, 0.17568798048992212, 0.20419331312810754, 0.26038989988497624, 0.22631161612482287]
# T=1000
#mean_rmse_values_FIVE[2] = 1*np.ones(N_lambdas)
#sosq_values_FIVE[2] = 1*np.ones(N_lambdas)
#mean_rmse_values_FIVE[2] = [3.219877687264644, 3.219877687264644, 2.6178999236796257, 2.320482048678608, 2.068685983887247, 1.2852522292869926, 0.8994032216808627, 0.746546579912313, 1.1729691308972325, 1.2578503608595315, 1.288263110054407, 1.2578148219878593, ]
#sosq_values_FIVE[2] = [6.24987472140568, 6.24987472140568, 0.07801970132136632, 0.05334935802899949, 0.11430684438529927, 0.8952474247685094, 0.11032693575986834, 0.9616738036477659, 0.6787680285322231, 0.2445739662225876, 0.2957435480200539, 0.09990701337324481, ]
# T=2000
#mean_rmse_values_FIVE[3] = 2*np.ones(N_lambdas)
#sosq_values_FIVE[3] = 2*np.ones(N_lambdas)
# T=3000
#mean_rmse_values_FIVE[4] = 3*np.ones(N_lambdas)
#sosq_values_FIVE[4] = 3*np.ones(N_lambdas)
# T=5000
#mean_rmse_values_FIVE[5] = []
#sosq_values_FIVE[5] = []

peak_lambda_array = [baseline_lambda_value + tuning_difference_array[i] for i in finished_lambda_indexes]
colors = [plt.cm.viridis(t) for t in np.linspace(0, 0.8, N_T_values)]

plt.figure()
plt.title("Average RMSE with background noise " + str(baseline_lambda_value))
plt.xlabel("Expected number of spikes in a bin at peak of tuning")
plt.ylabel("RMSE")
for T_index in range(N_T_values):
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_FIVE[T_index], yerr=(2.093*(np.sqrt(sosq_values_FIVE[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt="-", label="T="+str(T_array[T_index]), color = colors[T_index])
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_L[T_index], yerr=(2.093*(np.sqrt(sosq_values_L[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt="-", label="T="+str(T_array[T_index]), color = colors[T_index])
    plt.errorbar(x=peak_lambda_array, y=mean_rmse_values_RMSE[T_index], yerr=(2.093*(np.sqrt(sosq_values_RMSE[T_index]/degrees_of_freedom))/np.sqrt(N_seeds)), fmt="-", label="T="+str(T_array[T_index]), color = colors[T_index])
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xticks(range(11))
plt.vlines(baseline_lambda_value, 0, 3.8, label="Background tuning level")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness.png")
plt.show()


"""
#T = 500:
mean_rmse_values_FIVE[2,:] = []
sosq_values_FIVE[2,:] = []
#T = 1000:
mean_rmse_values_FIVE[2,:] = []
sosq_values_FIVE[2,:] = []
#T = 2000:
mean_rmse_values_FIVE[2,:] = []
sosq_values_FIVE[2,:] = []
#T = 3000:
mean_rmse_values_FIVE[2,:] = []
sosq_values_FIVE[2,:] = []
#T = 5000:
mean_rmse_values_FIVE[2,:] = []
sosq_values_FIVE[2,:] = []






# Mean values 
#T_10 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-10-up-to-lambda-9.5.npy")
#T_100 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-100-up-to-lambda-9.5.npy")
T_200 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-200-up-to-lambda-9.5.npy")
#T_500 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-500-up-to-lambda-9.5.npy")
#T_1000 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-1000-up-to-lambda-9.5.npy")
#T_2000 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-2000-up-to-lambda-9.5.npy")
#T_3162 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-3000-up-to-lambda-9.5.npy")
#T_5000 = np.load("mean_rmse_values_FIVE-base-lambda-0.5T-5000-up-to-lambda-9.5.npy")

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

#T_10 = np.load("mean_rmse_values_FIVE-T-10-up-to-lambda-10.npy")
#T_100 = np.load("mean_rmse_values_FIVE-T-100-up-to-lambda-10.npy")
#T_200 = np.load("mean_rmse_values_FIVE-T-200-up-to-lambda-10.npy")
#T_500 = np.load("mean_rmse_values_FIVE-T-500-up-to-lambda-10.npy")
#T_1000 = np.load("mean_rmse_values_FIVE-T-1000-up-to-lambda-10.npy")
#T_2000 = np.load("mean_rmse_values_FIVE-T-2000-up-to-lambda-5.npy")
#T_3162 = np.load("mean_rmse_values_FIVE-T-3162-up-to-lambda-10.npy")
#T_5000 = np.load("mean_rmse_values_FIVE-T-5000-up-to-lambda-10.npy")
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
