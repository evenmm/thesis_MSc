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

# These were averaged over seeds 10 to 21 only:
#lambda_strength_array = [0.1,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#rmse_T_100 = [1.3516777428732878, 1.374103612311344, 0.9979873504618402, 0.664476669684567, 0.5052838534861236, 0.39196103999443493, 0.31258698110538047, 0.3064841110552651, 0.2819506804468758, 0.23152420936810028, 0.23715513438121905, 0.35147284672781914, 0.42241686551823004, 0.4336790771343397, 0.3153930706565663, 0.22196219143972665, 0.2810792376898676, 0.4]
#plt.plot(lambda_strength_array, rmse_T_100, ".")

##a = np.load("mean_rmse_values-T-10-lambda-0.1.npy") # Stores lambdas up to this value
#a# = np.load("mean_rmse_values-T-10-up-to-lambda-15.npy") # Stores lambdas up to this value
#print(a)
#plt.plot(lambda_strength_array, a, ".")

#lambda_strength_array = [0.01,0.1,0.3,0.5,0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
lambda_strength_array = [0.01,0.1,0.3,0.5,0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
T_10 = np.load("mean_rmse_values-T-10-up-to-lambda-15.npy")
T_100 = np.load("mean_rmse_values-T-100-up-to-lambda-15.npy")
T_1000 = np.load("mean_rmse_values-T-1000-up-to-lambda-0.7.npy")

plt.figure()
plt.xlabel("Tuning strength")
plt.ylabel("RMSE")
plt.plot(lambda_strength_array, T_10, ".", label="T=10")
plt.plot(lambda_strength_array, T_100, ".", label="T=100")
plt.plot(lambda_strength_array, T_1000, ".", label="T=1000")
plt.legend(loc="upper right")
plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-robustness.png")
plt.show()
