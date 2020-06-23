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
from multiprocessing import Pool
from sklearn.decomposition import PCA

############################
# Parameters for inference #
############################
T = 1000 # change only after previous job is definitely RUNNING on cluster 
N_iterations = 20

global_initial_sigma_n = 2.5 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.95 # 0.99 # Learning rate by which we multiply sigma_n at every iteration

INFER_F_POSTERIORS = False
GRADIENT_FLAG = True # Set True to use analytic gradient
USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION = False
USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE = True
TOLERANCE = 1e-6
X_initialization = "ones" #"true" "ones" "pca" "randomrandom" "randomprior" "linspace"
# Using ensemble of PCA values
ensemble_smoothingwidths = [3,5,10] # [1,2,3,5,10,15,20,25,30,40,50,60,70]
LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE = False
FLIP_AFTER_SOME_ITERATION = False
FLIP_AFTER_HOW_MANY = 1
NOISE_REGULARIZATION = False
SMOOTHING_REGULARIZATION = False
GIVEN_TRUE_F = False
SPEEDCHECK = False
OPTIMIZE_HYPERPARAMETERS = False
PLOTTING = True
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
TUNINGCURVE_DEFINITION = "bumps" # "triangles" "bumps"
UNIFORM_BUMPS = False
PLOT_GRADIENT_CHECK = False
N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 40 # Number of grid points for plotting f posterior only 
tuning_width_delta = 1.2 # 0.1
# Peak lambda should not be defined as less than baseline h value
global peak_lambda_global
peak_lambda_global = 8
baseline_lambda_value = 0.5
baseline_f_value = np.log(baseline_lambda_value)
seeds = [11] #[0,2,3,4,5,6,8,9,11,12,16,17,18,19,21,22,25,26,28,29] # chosen only so that they cover the entire domain of X for T>=200 and sigma_x=40
NUMBER_OF_SEEDS = len(seeds)
sigma_f_fit = 2 #8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.83 # sqrt(0.7) # Scale for the tuning curve GP that is fitted. 0.3
# Define max and min of neural tuning 
lower_domain_limit = 0
upper_domain_limit = 10
how_many_added_neurons_outside_factor = 0.0 # Just makes it worse. If you must, use 0.1
min_neural_tuning_X = lower_domain_limit - how_many_added_neurons_outside_factor*(upper_domain_limit - lower_domain_limit)
max_neural_tuning_X = upper_domain_limit + how_many_added_neurons_outside_factor*(upper_domain_limit - lower_domain_limit)
min_inducing_point = lower_domain_limit
max_inducing_point = upper_domain_limit
# Neural density:
N = int((1+2*how_many_added_neurons_outside_factor)*100) # 100 with peaks in tuning area and 40 with tails coming in from each side
# For inference:
sigma_x = 40 # Variance of X for inference matrix K_t 
delta_x = 100 # Scale of X for inference matrix K_t
# Generative parameters for X path:
KEEP_PATH_INSIDE_DOMAIN_BY_FOLDING = True # Stop path from going outside defined domain with neurons
SCALE_UP_PATH_TO_COVER_DOMAIN = False # If True, the generated path is scaled up after being generated
sigma_x_generate_path = 40 # Variance for path generation. Set high enough so the path reaches max and min of tuning area
delta_x_generate_path = 100 # Scale for path generation.
jitter_term = 1e-5

print("-- using Example plotting parameter file --")
