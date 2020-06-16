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

# This bad boi branched off from em-algorithm on 11.05.2020
# and from robust-sim-data on 28.05.2020
# then from robust-efficient-script on 30.05.2020
## Set T
## For every lambda strength: Run 10 seeds and average RMS


############################
# Parameters for inference #
############################
T = 1000 # Max time 85504
N_iterations = 20

global_initial_sigma_n = 2.5 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.95 # 0.99 # Learning rate by which we multiply sigma_n at every iteration

INFER_F_POSTERIORS = False
GRADIENT_FLAG = True # Set True to use analytic gradient
USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION = False
USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE = True
TOLERANCE = 1e-5
X_initialization = "pca" #"true" "ones" "pca" "randomrandom" "randomprior" "linspace"
# Using ensemble of PCA values
ensemble_smoothingwidths = [2,3,5,10,15] # [1,2,3,5,10,15,20,25,30,40,50,60,70]
LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE = False # If False, they stay at (min_inducing_point, max_inducing_point)
FLIP_AFTER_SOME_ITERATION = False
FLIP_AFTER_HOW_MANY = 1
NOISE_REGULARIZATION = False
SMOOTHING_REGULARIZATION = False
GIVEN_TRUE_F = False
SPEEDCHECK = False
OPTIMIZE_HYPERPARAMETERS = False
PLOTTING = False
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
TUNINGCURVE_DEFINITION = "bumps" # "triangles" "bumps"
UNIFORM_BUMPS = False
PLOT_GRADIENT_CHECK = False
N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 40 # Number of grid points for plotting f posterior only 
tuning_width_delta = 1.2 # 0.1
# Peak lambda should not be defined as less than baseline h value
baseline_lambda_value = 0.5
baseline_f_value = np.log(baseline_lambda_value)
peak_lambda_array = [1.01,1.1,1.2,1.3,1.4,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5,6,7,8,9,10] #[2]#[4] #[0.01,0.1,0.3,0.5,0.7,1,1.5,2,2.5,3,4,5,6,7,8,9,10]
seeds = range(20) #[11] #[0,11,12,13,17] ## [0,3,5,9,11,12,13,15,19,21] good, 17 mediocre for T=100  [0,11,12,13,17] good for T=1000    1,2,6,8,10,14,20 bad      7,16 mediocre
NUMBER_OF_SEEDS = len(seeds)
print("Number of seeds we average over:", NUMBER_OF_SEEDS)
sigma_f_fit = 2 #8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.7 #0.5 # Scale for the tuning curve GP that is fitted. 0.3
# Define max and min of neural tuning 
min_inducing_point = 0
max_inducing_point = 10
min_neural_tuning_X = min_inducing_point
max_neural_tuning_X = max_inducing_point
# Neural density:
N = 100 #3*int(max_neural_tuning_X - min_neural_tuning_X) 
# For inference:
sigma_x = 20 # Variance of X for inference matrix K_t 
delta_x = 100 # Scale of X for inference matrix K_t
# Generative parameters for X path:
LIMIT_X_RANGE_AND_SCALE_TO_COVER_DOMAIN = True # Stop path from going outside defined domain with neurons
sigma_x_generate_path = 30 # Variance for path generation. Set high enough so the path reaches max and min of tuning area
delta_x_generate_path = 100 # Scale for path generation.

print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("Using gradient?", GRADIENT_FLAG)
print("Noise regulation:",NOISE_REGULARIZATION)
print("Initial sigma_n:", global_initial_sigma_n)
print("Learning rate:", lr)
print("T:", T)
print("N:", N)
print("Smoothingwidths:", ensemble_smoothingwidths)
if FLIP_AFTER_SOME_ITERATION:
    print("NBBBB!!! We're flipping the estimate in line 600.")
print("\n")
######################
# Covariance kernels #
######################

def squared_exponential_covariance(xvector1, xvector2, sigma, delta):
    if COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = scipy.spatial.distance.cdist(xvector1, xvector2, 'sqeuclidean')
    if COVARIANCE_KERNEL_KX == "periodic":
        # This handles paths that stretches across anywhere as though the domain is truly periodic
        # First put every time point between 0 and 2pi
        xvector1 = xvector1 % (2*np.pi)
        xvector2 = xvector2 % (2*np.pi)
        # Then take care of periodicity
        distancesquared_1 = scipy.spatial.distance.cdist(xvector1, xvector2, 'sqeuclidean')
        distancesquared_2 = scipy.spatial.distance.cdist(xvector1+2*np.pi, xvector2, 'sqeuclidean')
        distancesquared_3 = scipy.spatial.distance.cdist(xvector1-2*np.pi, xvector2, 'sqeuclidean')
        min_1 = np.minimum(distancesquared_1, distancesquared_2)
        distancesquared = np.minimum(min_1, distancesquared_3)
        #distancesquared = np.amin( [distancesquared_1, distancesquared_2, distancesquared_3] )
    return sigma * exp(-distancesquared/(2*delta))

def exponential_covariance(tvector1, tvector2, sigma, delta):
    absolutedistance = scipy.spatial.distance.cdist(tvector1, tvector2, 'euclidean')
    return sigma * exp(-absolutedistance/delta)

def f_exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def f_gaussian_periodic_covariance(x1,x2, sigma, delta):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return sigma * exp(-distancesquared/(2*delta))

def f_gaussian_NONPERIODIC_covariance(x1,x2, sigma, delta):
    distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

########################
# Covariance matrices  #
########################
K_t = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)

#########################
## Likelihood functions #
#########################

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize. Equation (4.17)++
def f_loglikelihood_bernoulli(f_i, sigma_n, y_i, K_xg_prev, K_gg): # Psi
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.log(1+np.exp(f_i))) # Corrected 16.03 from sum( np.multiply(y_i, (f_i - np.log(1+np.exp(f_i)))) + np.multiply((1-y_i), np.log(1- np.divide(np.exp(f_i), 1 + np.exp(f_i)))))
    priorterm_1 = -0.5*sigma_n**-2 * np.dot(f_i.T, f_i)
    fT_k = np.dot(f_i, K_xg_prev)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = 0.5*sigma_n**-2 * np.dot(np.dot(fT_k, smallinverse), fT_k.T)
    return - (likelihoodterm + priorterm_1 + priorterm_2)
def f_jacobian_bernoulli(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    yf_term = y_i - np.divide(np.exp(f_i), 1 + np.exp(f_i))
    priorterm_1 = -sigma_n**-2 * f_i
    kTf = np.dot(K_xg_prev.T, f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, kTf))
    f_derivative = yf_term + priorterm_1 + priorterm_2
    return - f_derivative
def f_hessian_bernoulli(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    e_tilde = np.divide(np.exp(f_i), (1 + np.exp(f_i))**2)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    f_hessian = - np.diag(e_tilde) - sigma_n**-2 * np.identity(T) + sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, K_xg_prev.T))
    return - f_hessian

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize.
def f_loglikelihood_poisson(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.exp(f_i)) 
    priorterm_1 = -0.5*sigma_n**-2 * np.dot(f_i.T, f_i)
    fT_k = np.dot(f_i, K_xg_prev)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = 0.5*sigma_n**-2 * np.dot(np.dot(fT_k, smallinverse), fT_k.T)
    return - (likelihoodterm + priorterm_1 + priorterm_2)

def f_jacobian_poisson(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    yf_term = y_i - np.exp(f_i)
    priorterm_1 = -sigma_n**-2 * f_i
    kTf = np.dot(K_xg_prev.T, f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, kTf))
    f_derivative = yf_term + priorterm_1 + priorterm_2
    return - f_derivative
def f_hessian_poisson(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    e_poiss = np.exp(f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    f_hessian = - np.diag(e_poiss) - sigma_n**-2*np.identity(T) + sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, K_xg_prev.T))
    return - f_hessian

# L function
def x_posterior_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce): 
    start = time.time()
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T
    stop = time.time()
    if SPEEDCHECK:
        print("Speedcheck of L function:")
        print("Making Kxg            :", stop-start)

    start = time.time()
    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    stop = time.time()
    if SPEEDCHECK:
        print("Making small/tempmatrx:", stop-start)

    # yf_term ##########
    ####################
    #start = time.time()
    #if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
    #    yf_term = sum(np.multiply(y_spikes, F_estimate) - np.log(1 + np.exp(F_estimate)))
    #elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
    #    yf_term = sum(np.multiply(y_spikes, F_estimate) - np.exp(F_estimate))
    #stop = time.time()
    #if SPEEDCHECK:
    #    print("yf term               :", stop-start)

    # f prior term #####
    ####################
    start = time.time()
    f_prior_term_1 = sigma_n**-2 * np.trace(np.matmul(F_estimate, F_estimate.T))
    fK = np.matmul(F_estimate, tempmatrix)
    fKf = np.matmul(fK, F_estimate.T)
    f_prior_term_2 = - sigma_n**-2 * np.trace(fKf)

    f_prior_term = - 0.5 * (f_prior_term_1 + f_prior_term_2)
    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)

    # logdet term ######
    ####################
    # My variant: 
    #logdet_term = - 0.5 * N * np.log(np.linalg.det(Kx_inducing))
    # Wu variant:
    start = time.time()
    logDetS1 = -np.log(np.linalg.det(smallinverse))-np.log(np.linalg.det(K_gg))+np.log(sigma_n)*(T-N_inducing_points)
    logdet_term = - 0.5 * N * logDetS1
    stop = time.time()
    if SPEEDCHECK:
        print("logdet term            :", stop-start)

    # x prior term #####
    ####################
    start = time.time()
    xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)
    stop = time.time()
    if SPEEDCHECK:
        print("X prior term          :", stop-start)
        print("logdet_term", logdet_term)
        print("f_prior_term", f_prior_term)
        print("x_prior_term", x_prior_term)
    posterior_loglikelihood = logdet_term + f_prior_term + x_prior_term #+ yf_term
    return - posterior_loglikelihood

# Gradient of L 
def x_jacobian_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    ####################
    # Initial matrices #
    ####################
    start = time.time()
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T
    stop = time.time()
    if SPEEDCHECK:
        print("\nSpeedcheck of x_jacobian function:")
        print("Making Kxg            :", stop-start)

    start = time.time()
    B_matrix = np.matmul(K_gx, K_xg) + (sigma_n**2) * K_gg
    B_matrix_inverse = np.linalg.inv(B_matrix)
    stop = time.time()
    if SPEEDCHECK:
        print("Making B and B inverse:", stop-start)

    start = time.time()
    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    #smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    stop = time.time()
    if SPEEDCHECK:
        print("Making small/tempmatrx:", stop-start)

    ####################
    # logdet term ######
    ####################
    start = time.time()

    ## Evaluate the derivative of K_xg. Row t of this matrix holds the nonzero row of the matrix d/dx_t K_xg
    d_Kxg = scipy.spatial.distance.cdist(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), lambda u, v: -(u-v)*np.exp(-(u-v)**2/(2*delta_f_fit**2)))
    d_Kxg = d_Kxg*sigma_f_fit*(delta_f_fit**-2)

    ## Reshape K_gx and K_xg to speed up matrix multiplication
    K_g_column_tensor = K_gx.T.reshape((T, N_inducing_points, 1)) # Tensor with T depth containing single columns of length N_ind 
    d_Kx_row_tensor = d_Kxg.reshape((T, 1, N_inducing_points)) # Tensor with T depth containing single rows of length N_ind 

    # Matrix multiply K_gx and d(K_xg)
    product_Kgx_dKxg = np.matmul(K_g_column_tensor, d_Kx_row_tensor) # 1000 by 30 by 30

    # Sum with transpose
    trans_sum_K_dK = product_Kgx_dKxg + np.transpose(product_Kgx_dKxg, axes=(0,2,1))

    # Create B^-1 copies for vectorial matrix multiplication
    B_inv_tensor = np.repeat([B_matrix_inverse],T,axis=0)
    
    # Then tensor multiply B^-1 with all the different trans_sum_K_dK
    big_tensor = np.matmul(B_inv_tensor, trans_sum_K_dK)
    
    # Take trace of each individually
    trace_array = np.trace(big_tensor, axis1=1, axis2=2)
    
    # Multiply by - N/2
    logdet_gradient = - N/2 * trace_array

    stop = time.time()
    if SPEEDCHECK:
        print("logdet term            :", stop-start)

    ####################
    # f prior term ##### (speeded up 10x)
    ####################
    start = time.time()
    fMf = np.zeros((T,N,N))

    ## New hot take:
    # Elementwise in the sum, priority on things with dim T, AND things that don't need to be vectorized *first*.
    # Wrap things in from the sides to sandwich the tensor.
    f_Kx = np.matmul(F_estimate, K_xg)
    f_Kx_Binv = np.matmul(f_Kx, B_matrix_inverse)
    #Binv_Kg_f = np.transpose(f_Kx_Binv)

    #d_Kg_column_tensor = np.transpose(d_Kx_row_tensor, axes=(0,2,1))

    # partial derivatives need tensorization
    # f_dKx = np.matmul(F_estimate, d_Kxg)
    f_column_tensor = F_estimate.T.reshape((T, N, 1))
    f_dKx_tensor = np.matmul(f_column_tensor, d_Kx_row_tensor) # (N x N_inducing) matrices  
    dKg_f_tensor = np.transpose(f_dKx_tensor, axes=(0,2,1))

    f_Kx_Binv_copy_tensor = np.repeat([f_Kx_Binv], T, axis=0)
    Binv_Kg_f_copy_tensor = np.transpose(f_Kx_Binv_copy_tensor, axes=(0,2,1)) #repeat([Binv_Kg_f], T, axis=0)

    ## A: f dKx Binv Kgx f
    fMf += np.matmul(f_dKx_tensor, Binv_Kg_f_copy_tensor)

    ## C: - f Kx Binv Kg dKx Binv Kg f
    Kg_dKx_tensor = np.matmul(K_g_column_tensor, d_Kx_row_tensor)
    f_Kx_Binv_Kg_dKx_tensor = np.matmul(f_Kx_Binv_copy_tensor, Kg_dKx_tensor)
    fMf -= np.matmul(f_Kx_Binv_Kg_dKx_tensor, Binv_Kg_f_copy_tensor)

    ## B: - f Kx Binv dKg Kx Binv Kg f
    dKg_Kx_tensor = np.transpose(Kg_dKx_tensor, axes=(0,2,1))
    f_Kx_Binv_dKg_Kx_tensor = np.matmul(f_Kx_Binv_copy_tensor, dKg_Kx_tensor)
    fMf -= np.matmul(f_Kx_Binv_dKg_Kx_tensor, Binv_Kg_f_copy_tensor)

    ## D: f Kx Binv dKg f
    fMf += np.matmul(f_Kx_Binv_copy_tensor, dKg_f_tensor)

    ## Trace for each matrix in the tensor
    fMfsum = np.trace(fMf, axis1=1, axis2=2)
    f_prior_gradient = sigma_n**(-2) / 2 * fMfsum

    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)

    ####################
    # x prior term #####
    ####################
    start = time.time()
    x_prior_gradient = (-1) * np.dot(X_estimate.T, K_t_inverse)
    stop = time.time()
    if SPEEDCHECK:
        print("X prior term          :", stop-start)
    ####################
    x_gradient = logdet_gradient + f_prior_gradient + x_prior_gradient 
    return - x_gradient

def just_fprior_term(X_estimate): 
    K_xg = np.zeros((T,N_inducing_points))
    for x1 in range(T):
        for x2 in range(N_inducing_points):
            K_xg[x1,x2] = squared_exponential_covariance(X_estimate[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T

    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)

    # yf_term ##########
    ####################
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, true_f) - np.log(1 + np.exp(true_f)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, true_f) - np.exp(true_f))

    # f prior term #####
    ####################
    f_prior_term_1 = sigma_n**-2 * np.trace(np.matmul(F_estimate, F_estimate.T))
    fK = np.matmul(F_estimate, tempmatrix)
    fKf = np.matmul(fK, F_estimate.T)
    f_prior_term_2 = - sigma_n**-2 * np.trace(fKf)

    f_prior_term = - 0.5 * (f_prior_term_1 + f_prior_term_2)
    # logdet term ######
    ####################
    # My variant: 
    #logdet_term = - 0.5 * N * np.log(np.linalg.det(Kx_inducing))
    # Wu variant:
    logDetS1 = -np.log(np.linalg.det(smallinverse))-np.log(np.linalg.det(K_gg))+np.log(sigma_n)*(T-N_inducing_points)
    logdet_term = - 0.5 * N * logDetS1

    # x prior term #####
    ####################
    #xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    #x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)

    #print("f_prior_term",f_prior_term)
    #print("logdet_term",logdet_term)
    #print("x_prior_term",x_prior_term)
    posterior_loglikelihood = yf_term + f_prior_term #+ logdet_term #+ x_prior_term
    return - posterior_loglikelihood

######################################
## Data generation                  ##
######################################
bins_for_plotting = np.linspace(min_neural_tuning_X, max_neural_tuning_X, num=N_plotgridpoints + 1)
x_grid_for_plotting = 0.5*(bins_for_plotting[:(-1)]+bins_for_plotting[1:])
K_t_generate = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_x_generate_path, delta_x_generate_path)
if UNIFORM_BUMPS:
    # Uniform positioning and width:'
    bumplocations = [min_neural_tuning_X + (i+0.5)/N*(max_neural_tuning_X - min_neural_tuning_X) for i in range(N)]
    bump_delta_distances = tuning_width_delta * np.ones(N)
else:
    # Random placement and width:
    bumplocations = min_neural_tuning_X + (max_neural_tuning_X - min_neural_tuning_X) * np.random.random(N)
    bump_delta_distances = tuning_width_delta + tuning_width_delta/4*np.random.random(N)
def bumptuningfunction(x, i, peak_f_offset): 
    x1 = x
    x2 = bumplocations[i]
    delta_bumptuning = bump_delta_distances[i]
    if COVARIANCE_KERNEL_KX == "periodic":
        distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    elif COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = (x1-x2)**2
    return baseline_f_value + peak_f_offset * exp(-distancesquared/(2*delta_bumptuning))

def offset_function(offset_for_estimate, X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    offset_estimate = X_estimate + offset_for_estimate
    return x_posterior_no_la(offset_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)

def scaling_function(scaling_factor, X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    scaled_estimate = scaling_factor*X_estimate
    return x_posterior_no_la(scaled_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)

def scale_and_offset_function(scale_offset, X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    scaled_estimate = scale_offset[0] * X_estimate + scale_offset[1]
    return x_posterior_no_la(scaled_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
    #return just_fprior_term(scaled_estimate)

######################################
## RMSE function                    ##
######################################
def find_rmse_for_this_lambda_this_seed(seedindex):
    print("Seed", seeds[seedindex], "started.")
    peak_f_offset = np.log(peak_lambda_global) - baseline_f_value
    np.random.seed(seeds[seedindex])
    # Generate path
    path = (max_neural_tuning_X-min_neural_tuning_X)/2 + numpy.random.multivariate_normal(np.zeros(T), K_t_generate)
    #path = np.linspace(min_neural_tuning_X, max_neural_tuning_X, T)
    if LIMIT_X_RANGE_AND_SCALE_TO_COVER_DOMAIN:
        # Use boolean masks to keep X within min and max of tuning 
        path -= min_neural_tuning_X # bring path to 0
        modulo_two_pi_values = path // (max_neural_tuning_X)
        oddmodulos = (modulo_two_pi_values % 2).astype(bool)
        evenmodulos = np.invert(oddmodulos)
        # Even modulos: Adjust for being outside
        path[evenmodulos] -= max_neural_tuning_X*modulo_two_pi_values[evenmodulos]
        # Odd modulos: Adjust for being outside and flip for continuity
        path[oddmodulos] -= max_neural_tuning_X*(modulo_two_pi_values[oddmodulos])
        differences = max_neural_tuning_X - path[oddmodulos]
        path[oddmodulos] = differences
        path += min_neural_tuning_X # bring path back to min value for tuning
        # Now scale to cover the domain:
        path -= min(path)
        path /= max(path)
        path *= (max_inducing_point-min_inducing_point)
        path += min_inducing_point
    if PLOTTING:
        ## plot path 
        if T > 100:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        plt.plot(path, color="black", label='True X') #plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
        #plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.title("True path of X")
        plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-T-" + str(T)  + "-seed-" + str(seeds[seedindex]) + "-path.png")
    ## Generate spike data. True tuning curves are defined here
    if TUNINGCURVE_DEFINITION == "triangles":
        tuningwidth = 1 # width of tuning (in radians)
        biasterm = -2 # Average H outside tuningwidth -4
        tuningcovariatestrength = np.linspace(0.5*tuningwidth,10.*tuningwidth, N) # H value at centre of tuningwidth 6*tuningwidth
        neuronpeak = [min_neural_tuning_X + (i+0.5)/N*(max_neural_tuning_X - min_neural_tuning_X) for i in range(N)]
        true_f = np.zeros((N, T))
        y_spikes = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                if COVARIANCE_KERNEL_KX == "periodic":
                    distancefrompeaktopathpoint = min([ abs(neuronpeak[i]+2.*pi-path[t]),  abs(neuronpeak[i]-path[t]),  abs(neuronpeak[i]-2.*pi-path[t]) ])
                elif COVARIANCE_KERNEL_KX == "nonperiodic":
                    distancefrompeaktopathpoint = abs(neuronpeak[i]-path[t])
                Ht = biasterm
                if(distancefrompeaktopathpoint < tuningwidth):
                    Ht = biasterm + tuningcovariatestrength[i] * (1-distancefrompeaktopathpoint/tuningwidth)
                true_f[i,t] = Ht
                # Spiking
                if LIKELIHOOD_MODEL == "bernoulli":
                    spike_probability = exp(Ht)/(1.+exp(Ht))
                    y_spikes[i,t] = 1.0*(rand()<spike_probability)
                    # If you want to remove randomness: y_spikes[i,t] = spike_probability
                elif LIKELIHOOD_MODEL == "poisson":
                    spike_rate = exp(Ht)
                    y_spikes[i,t] = np.random.poisson(spike_rate)
                    # If you want to remove randomness: y_spikes[i,t] = spike_rate
    elif TUNINGCURVE_DEFINITION == "bumps":
        true_f = np.zeros((N, T))
        y_spikes = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                true_f[i,t] = bumptuningfunction(path[t], i, peak_f_offset)
                if LIKELIHOOD_MODEL == "bernoulli":
                    spike_probability = exp(true_f[i,t])/(1.+exp(true_f[i,t]))
                    y_spikes[i,t] = 1.0*(rand()<spike_probability)
                elif LIKELIHOOD_MODEL == "poisson":
                    spike_rate = exp(true_f[i,t])
                    y_spikes[i,t] = np.random.poisson(spike_rate)
    if PLOTTING:
        ## Plot true f in time
        plt.figure()
        color_idx = np.linspace(0, 1, N)
        plt.title("True F")
        plt.xlabel("x")
        plt.ylabel("f value")
        x_space_grid = np.linspace(min_neural_tuning_X, max_neural_tuning_X, T)
        for i in range(N):
            plt.plot(x_space_grid, true_f[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-true-f.png")
        #plt.show()
    ###############################
    # Covariance matrix Kgg_plain #
    ###############################
    # Inducing points based on the actual range of X
    x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points) #np.linspace(min(path), max(path), N_inducing_points)
    #print("Min and max of path:", min(path), max(path))
    #print("Min and max of grid:", min(x_grid_induce), max(x_grid_induce))
    K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    ######################
    # Initialize X and F #
    ######################
    # Here the PCA ensemble comes into play:
    ensemble_array_X_rmse = np.zeros(len(ensemble_smoothingwidths))
    ensemble_array_X_estimate = np.zeros((len(ensemble_smoothingwidths), T))
    ensemble_array_F_estimate = np.zeros((len(ensemble_smoothingwidths), N, T))
    ensemble_array_y_spikes = np.zeros((len(ensemble_smoothingwidths), N, T))
    ensemble_array_path = np.zeros((len(ensemble_smoothingwidths), T))
    for smoothingwindow_index in range(len(ensemble_smoothingwidths)):
        smoothingwindow_for_PCA = ensemble_smoothingwidths[smoothingwindow_index]
        # PCA initialization: 
        celldata = zeros(shape(y_spikes))
        for i in range(N):
            celldata[i,:] = scipy.ndimage.filters.gaussian_filter1d(y_spikes[i,:], smoothingwindow_for_PCA) # smooth
            #celldata[i,:] = (celldata[i,:]-mean(celldata[i,:]))/std(celldata[i,:])                 # standardization requires at least one spike
        X_pca_result = PCA(n_components=1, svd_solver='full').fit_transform(transpose(celldata))
        X_pca_initial = np.zeros(T)
        for i in range(T):
            X_pca_initial[i] = X_pca_result[i]
        # Scale PCA initialization to fit domain:
        X_pca_initial -= min(X_pca_initial)
        X_pca_initial /= max(X_pca_initial)
        X_pca_initial *= (max_inducing_point-min_inducing_point)
        X_pca_initial += min_inducing_point
        # Flip PCA initialization correctly by comparing to true path
        X_pca_initial_flipped = 2*mean(X_pca_initial) - X_pca_initial
        X_pca_initial_rmse = np.sqrt(sum((X_pca_initial-path)**2) / T)
        X_pca_initial_flipped_rmse = np.sqrt(sum((X_pca_initial_flipped-path)**2) / T)
        if X_pca_initial_flipped_rmse < X_pca_initial_rmse:
            X_pca_initial = X_pca_initial_flipped
        # Plot PCA initialization
        if T > 100:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.title("PCA initial of X")
        plt.plot(path, color="black", label='True X')
        plt.plot(X_pca_initial, label="Initial")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-PCA-initial.png")

        # Initialize X
        np.random.seed(0)
        if X_initialization == "true":
            X_initial = path
        if X_initialization == "ones":
            X_initial = np.ones(T)
        if X_initialization == "pca":
            X_initial = X_pca_initial
        if X_initialization == "randomrandom":
            X_initial = (max_neural_tuning_X - min_neural_tuning_X)*np.random.random(T)
        if X_initialization == "randomprior":
            X_initial = (max_neural_tuning_X - min_neural_tuning_X)*np.random.multivariate_normal(np.zeros(T), K_t_generate)
        if X_initialization == "linspace":
            X_initial = np.linspace(min_neural_tuning_X, max_neural_tuning_X, T) 
        X_estimate = np.copy(X_initial)
        # Initialize F
        F_initial = np.sqrt(y_spikes) - np.amax(np.sqrt(y_spikes))/2 #np.log(y_spikes + 0.0008)
        F_estimate = np.copy(F_initial)
        if GIVEN_TRUE_F:
            F_estimate = true_f
        if PLOTTING:
            if T > 100:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            #plt.title("Path of X")
            plt.title("X estimate")
            plt.xlabel("Time")
            plt.ylabel("x")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_initial, label='Initial')
            #plt.legend(loc="upper right")
            #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + ".png")
        if PLOT_GRADIENT_CHECK:
            sigma_n = np.copy(global_initial_sigma_n)
            K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
            X_gradient = x_jacobian_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
            if T > 100:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            plt.xlabel("Time")
            plt.ylabel("x")
            plt.title("Gradient at initial X")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_initial, label="Initial")
            #plt.plot(X_gradient, label="Gradient")
            plt.plot(X_estimate + 2*X_gradient/max(X_gradient), label="Gradient plus offset")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-Gradient.png")
            exit()
            """
            print("Testing gradient...")
            #X_estimate = path
            #F_estimate = true_f
            print("Gradient difference using check_grad:",scipy.optimize.check_grad(func=x_posterior_no_la, grad=x_jacobian_no_la, x0=path, args=(sigma_n, F_estimate, K_gg, x_grid_induce)))

            #optim_gradient = optimization_result.jac
            print("Epsilon:", np.sqrt(np.finfo(float).eps))
            optim_gradient1 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=1*np.sqrt(np.finfo(float).eps), args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            optim_gradient2 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=x_posterior_no_la, 1e-4, args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            optim_gradient3 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=x_posterior_no_la, 1e-2, args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            optim_gradient4 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=x_posterior_no_la, 1e-2, args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            calculated_gradient = x_jacobian_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
            difference_approx_fprime_1 = optim_gradient1 - calculated_gradient
            difference_approx_fprime_2 = optim_gradient2 - calculated_gradient
            difference_approx_fprime_3 = optim_gradient3 - calculated_gradient
            difference_approx_fprime_4 = optim_gradient4 - calculated_gradient
            difference_norm1 = np.linalg.norm(difference_approx_fprime_1)
            difference_norm2 = np.linalg.norm(difference_approx_fprime_2)
            difference_norm3 = np.linalg.norm(difference_approx_fprime_3)
            difference_norm4 = np.linalg.norm(difference_approx_fprime_4)
            print("Gradient difference using approx f prime, epsilon 1e-8:", difference_norm1)
            print("Gradient difference using approx f prime, epsilon 1e-4:", difference_norm2)
            print("Gradient difference using approx f prime, epsilon 1e-2:", difference_norm3)
            print("Gradient difference using approx f prime, epsilon 1e-2:", difference_norm4)
            plt.figure()
            plt.title("Gradient compared to numerical gradient")
            plt.plot(calculated_gradient, label="Analytic")
            #plt.plot(optim_gradient1, label="Numerical 1")
            plt.plot(optim_gradient2, label="Numerical 2")
            plt.plot(optim_gradient3, label="Numerical 3")
            plt.plot(optim_gradient4, label="Numerical 4")
            plt.legend()
            plt.figure()
            #plt.plot(difference_approx_fprime_1, label="difference 1")
            plt.plot(difference_approx_fprime_2, label="difference 2")
            plt.plot(difference_approx_fprime_3, label="difference 3")
            plt.plot(difference_approx_fprime_4, label="difference 4")
            plt.legend()
            plt.show()
            exit()
            """
        #############################
        # Iterate with EM algorithm #
        #############################
        prev_X_estimate = np.Inf
        sigma_n = np.copy(global_initial_sigma_n)
        for iteration in range(N_iterations):
            if iteration > 0:
                sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
                if LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE:
                    x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) # Change position of grid to position of estimate
            K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
            K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
            # Find F estimate only if we're not at the first iteration
            if iteration > 0:
                if LIKELIHOOD_MODEL == "bernoulli":
                    for i in range(N):
                        y_i = y_spikes[i]
                        optimization_result = optimize.minimize(fun=f_loglikelihood_bernoulli, x0=F_estimate[i], jac=f_jacobian_bernoulli, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
                        F_estimate[i] = optimization_result.x
                elif LIKELIHOOD_MODEL == "poisson":
                    for i in range(N):
                        y_i = y_spikes[i]
                        optimization_result = optimize.minimize(fun=f_loglikelihood_poisson, x0=F_estimate[i], jac=f_jacobian_poisson, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
                        F_estimate[i] = optimization_result.x 
            # Find next X estimate, that can be outside (0,2pi)
            if GIVEN_TRUE_F: 
                print("NB! NB! We're setting the f value to the optimal F given the path.")
                F_estimate = np.copy(true_f)
            if NOISE_REGULARIZATION:
                X_estimate += 2*np.random.multivariate_normal(np.zeros(T), K_t_generate) - 1
            if SMOOTHING_REGULARIZATION and iteration < (N_iterations-1) :
                X_estimate = scipy.ndimage.filters.gaussian_filter1d(X_estimate, 4)
            if GRADIENT_FLAG: 
                optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
            else:
                optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
            X_estimate = optimization_result.x
            if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
                # Flipping estimate after iteration 1 has been plotted
                X_estimate = 2*mean(X_estimate) - X_estimate
            if USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION:
                X_estimate -= min(X_estimate) #set offset of min to 0
                X_estimate /= max(X_estimate) #scale length to 1
                X_estimate *= (max(path)-min(path)) #scale length to length of path
                X_estimate += min(path) #set offset to offset of path
            if PLOTTING:
                plt.plot(X_estimate, label='Estimate')
                #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
                plt.tight_layout()
                plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + ".png")
            if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
                #print("Seed", seeds[seedindex], "Iterations:", iteration+1, "Change in X smaller than TOL")
                break
            #if iteration == N_iterations-1:
            #    print("Seed", seeds[seedindex], "Iterations:", iteration+1, "N_iterations reached")
            prev_X_estimate = X_estimate
        if USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE:
            X_estimate -= min(X_estimate) #set offset of min to 0
            X_estimate /= max(X_estimate) #scale length to 1
            X_estimate *= (max(path)-min(path)) #scale length to length of path
            X_estimate += min(path) #set offset to offset of path
        # Flipped 
        X_flipped = - X_estimate + 2*mean(X_estimate)
        # Rootmeansquarederror for X
        X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
        X_flipped_rmse = np.sqrt(sum((X_flipped-path)**2) / T)
        ##### Check if flipped and maybe iterate again with flipped estimate
        if X_flipped_rmse < X_rmse:
            #print("RMSE for X:", X_rmse)
            #print("RMSE for X flipped:", X_flipped_rmse)
            #print("Re-iterating because of flip")
            x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points) #np.linspace(min(path), max(path), N_inducing_points)
            K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
            X_initial_2 = np.copy(X_flipped)
            X_estimate = np.copy(X_flipped)
            F_estimate = np.copy(F_initial)
            if GIVEN_TRUE_F:
                F_estimate = true_f
            if PLOTTING:
                if T > 100:
                    plt.figure(figsize=(10,3))
                else:
                    plt.figure()
                #plt.title("After flipping") # as we go
                plt.xlabel("Time")
                plt.ylabel("x")
                plt.plot(path, color="black", label='True X')
                plt.plot(X_initial_2, label='Initial')
                #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
                plt.tight_layout()
                plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-flipped.png")
            #############################
            # EM after flipped          #
            #############################
            prev_X_estimate = np.Inf
            sigma_n = np.copy(global_initial_sigma_n)
            for iteration in range(N_iterations):
                if iteration > 0:
                    sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
                    if LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE:
                        x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) # Change position of grid to position of estimate
                K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
                K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
                # Find F estimate only if we're not at the first iteration
                if iteration > 0:
                    if LIKELIHOOD_MODEL == "bernoulli":
                        for i in range(N):
                            y_i = y_spikes[i]
                            optimization_result = optimize.minimize(fun=f_loglikelihood_bernoulli, x0=F_estimate[i], jac=f_jacobian_bernoulli, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
                            F_estimate[i] = optimization_result.x
                    elif LIKELIHOOD_MODEL == "poisson":
                        for i in range(N):
                            y_i = y_spikes[i]
                            optimization_result = optimize.minimize(fun=f_loglikelihood_poisson, x0=F_estimate[i], jac=f_jacobian_poisson, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
                            F_estimate[i] = optimization_result.x 
                # Find next X estimate, that can be outside (0,2pi)
                if GIVEN_TRUE_F: 
                    print("NB! NB! We're setting the f value to the optimal F given the path.")
                    F_estimate = np.copy(true_f)
                if NOISE_REGULARIZATION:
                    X_estimate += 2*np.random.multivariate_normal(np.zeros(T), K_t_generate) - 1
                if SMOOTHING_REGULARIZATION and iteration < (N_iterations-1) :
                    X_estimate = scipy.ndimage.filters.gaussian_filter1d(X_estimate, 4)
                if GRADIENT_FLAG: 
                    optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
                else:
                    optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
                X_estimate = optimization_result.x
                if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
                    # Flipping estimate after iteration 1 has been plotted
                    X_estimate = 2*mean(X_estimate) - X_estimate
                if USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION:
                    X_estimate -= min(X_estimate) #set offset of min to 0
                    X_estimate /= max(X_estimate) #scale length to 1
                    X_estimate *= (max(path)-min(path)) #scale length to length of path
                    X_estimate += min(path) #set offset to offset of path
                if PLOTTING:
                    plt.plot(X_estimate, label='Estimate (after flip)')
                    #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
                    plt.tight_layout()
                    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-flipped.png")
                if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
                    #print("Seed", seeds[seedindex], "Iterations after flip:", iteration+1, "Change in X smaller than TOL")
                    break
                #if iteration == N_iterations-1:
                #    print("Seed", seeds[seedindex], "Iterations after flip:", iteration+1, "N_iterations reached")
                prev_X_estimate = X_estimate
            if USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE:
                X_estimate -= min(X_estimate) #set offset of min to 0
                X_estimate /= max(X_estimate) #scale length to 1
                X_estimate *= (max(path)-min(path)) #scale length to length of path
                X_estimate += min(path) #set offset to offset of path
            # Rootmeansquarederror for X
            X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
        #print("Seed", seeds[seedindex], "smoothingwindow", smoothingwindow_for_PCA, "finished. RMSE for X:", X_rmse)
        #SStot = sum((path - mean(path))**2)
        #SSdev = sum((X_estimate-path)**2)
        #Rsquared = 1 - SSdev / SStot
        #Rsquared_values[seed] = Rsquared
        #print("R squared value of X estimate:", Rsquared, "\n")
        #####
        # Rootmeansquarederror for F
        #if LIKELIHOOD_MODEL == "bernoulli":
        #    h_estimate = np.divide( np.exp(F_estimate), (1 + np.exp(F_estimate)))
        #if LIKELIHOOD_MODEL == "poisson":
        #    h_estimate = np.exp(F_estimate)
        #F_rmse = np.sqrt(sum((h_estimate-true_f)**2) / (T*N))
        if PLOTTING:
            if T > 100:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            plt.title("Final estimate") # as we go
            plt.xlabel("Time")
            plt.ylabel("x")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_initial, label='Initial')
            plt.plot(X_estimate, label='Estimate')
            plt.legend(loc="upper right")
            #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-final.png")
        ensemble_array_X_rmse[smoothingwindow_index] = X_rmse
        ensemble_array_X_estimate[smoothingwindow_index] = X_estimate
        ensemble_array_F_estimate[smoothingwindow_index] = F_estimate
        ensemble_array_y_spikes[smoothingwindow_index] = y_spikes
        ensemble_array_path[smoothingwindow_index] = path
        # Finish loop for one smoothingwidth
    # Find best rmse across smoothingwindows for PCA start:
    best_rmse_index = np.argmin(ensemble_array_X_rmse)
    X_rmse = ensemble_array_X_rmse[best_rmse_index]
    X_estimate = ensemble_array_X_estimate[best_rmse_index]
    F_estimate = ensemble_array_F_estimate[best_rmse_index]
    y_spikes = ensemble_array_y_spikes[best_rmse_index]
    path = ensemble_array_path[best_rmse_index]
    print("Seed", seeds[seedindex], "RMSEs", ensemble_array_X_rmse, "\nBest smoothing window:", ensemble_smoothingwidths[best_rmse_index], "with RMSE:", X_rmse)
    return [X_rmse, X_estimate, F_estimate, y_spikes, path]

seed_rmse_array = np.zeros(len(seeds))
X_array = np.zeros((len(seeds), T))
F_array = np.zeros((len(seeds), N, T))
Y_array = np.zeros((len(seeds), N, T))
path_array = np.zeros((len(seeds), T))
if __name__ == "__main__": 
    # We gather the mean rmse values for each tuning strength in this array:
    mean_rmse_values = np.zeros(len(peak_lambda_array))
    sum_of_squared_deviation_values = np.zeros(len(peak_lambda_array))
    for lambda_index in range(len(peak_lambda_array)):
        global peak_lambda_global
        peak_lambda_global = peak_lambda_array[lambda_index]

        # Pool computing
        print("Time to make a pool")
        starttime = time.time()
        myPool = Pool(processes=len(seeds))
        result_array = myPool.map(find_rmse_for_this_lambda_this_seed, [i for i in range(len(seeds))])
        myPool.close()
        endtime = time.time()

        # Unpack results
        for i in range(len(seeds)):
            seed_rmse_array[i] = result_array[i][0]
            X_array[i] = result_array[i][1]
            F_array[i] = result_array[i][2]
            Y_array[i] = result_array[i][3]
            path_array[i] = result_array[i][4]
        mean_rmse_values[lambda_index] = np.mean(seed_rmse_array)
        sum_of_squared_deviation_values[lambda_index] = sum((seed_rmse_array - np.mean(seed_rmse_array))**2)
        np.save("mean_rmse_values-base-lambda-" + baseline_lambda_value + "T-" + str(T) + "-up-to-lambda-" + str(peak_lambda_global), mean_rmse_values)
        np.save("sum_of_squared_deviation_values-base-lambda-" + baseline_lambda_value + "T-" + str(T) + "-up-to-lambda-" + str(peak_lambda_global), sum_of_squared_deviation_values)

        print("\n")
        print("Lambda strength:", peak_lambda_global)
        #print("Array of rmse for seeds:", seed_rmse_array)
        print("RMSE for X, Averaged across seeds:", mean_rmse_values[lambda_index])
        print("STD for RMSE:", sum_of_squared_deviation_values[lambda_index])
        print("Time use:", endtime - starttime)
        print("\n")

if not INFER_F_POSTERIORS:
    exit()

F_estimate = F_array[0]
sigma_n = 1
y_spikes = Y_array[0]
path = path_array[0] 
X_estimate = X_array[0]

#X_estimate = path
#print("Setting X_estimate = path for posterior F")

## X_estimate is used to make Kxg.
## A new grid is introduced here for plotting (not really necessary but it works)
#################################################
# Find posterior prediction of log tuning curve #
#################################################

# Inducing points (g efers to inducing points. Originally u did.)
x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points)

# K_xg = K_fu
K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
K_gx = K_xg.T

# K_gg = K_uu and means inducing points
K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
K_gg_inverse = np.linalg.inv(K_gg)

# Connect x to plotgrid through inducing points
K_g_plotgrid = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)
K_plotgrid_g = K_g_plotgrid.T

# Plot K_g_plotgrid
fig, ax = plt.subplots()
kx_cross_mat = ax.matshow(K_g_plotgrid, cmap=plt.cm.Blues)
fig.colorbar(kx_cross_mat, ax=ax)
plt.title("K_g_plotgrid")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-K_g_plotgrid.png")
print("Making spatial covariance matrice: Kx grid")

K_plotgrid_plotgrid = squared_exponential_covariance(x_grid_for_plotting.reshape((N_plotgridpoints,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)

# Plot K_plotgrid_plotgrid
fig, ax = plt.subplots()
kxmat = ax.matshow(K_plotgrid_plotgrid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.title("K_plotgrid_plotgrid")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-K_plotgrid_plotgrid.png")

Q_plotgrid_x = np.matmul(np.matmul(K_plotgrid_g, K_gg_inverse), K_gx)
Q_x_plotgrid = Q_plotgrid_x.T

# Infer mean on the grid
smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
Q_xx_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
Kxx_times_F = np.matmul(Q_xx_plus_sigma_inverse, F_estimate.T)
mu_posterior = np.matmul(Q_plotgrid_x, Kxx_times_F) # Here we have Kx crossover. Check what happens if swapped with Q = KKK

# Calculate standard deviations
sigma_posterior = K_plotgrid_plotgrid - np.matmul(Q_plotgrid_x, np.matmul(Q_xx_plus_sigma_inverse, Q_x_plotgrid))

# Plot posterior covariance matrix
fig, ax = plt.subplots()
sigma_posteriormat = ax.matshow(sigma_posterior, cmap=plt.cm.Blues)
fig.colorbar(sigma_posteriormat, ax=ax)
plt.title("Posterior covariance matrix")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-sigma_posterior.png")

###############################################
# Plot tuning curve with confidence intervals #
###############################################
standard_deviation = [np.sqrt(np.diag(sigma_posterior))]
print("posterior marginal standard deviation:\n",standard_deviation[0])
standard_deviation = np.repeat(standard_deviation, N, axis=0)
upper_confidence_limit = mu_posterior + 1.96*standard_deviation.T
lower_confidence_limit = mu_posterior - 1.96*standard_deviation.T

if LIKELIHOOD_MODEL == "bernoulli":
    h_estimate = np.divide( np.exp(mu_posterior), (1 + np.exp(mu_posterior)))
    h_upper_confidence_limit = np.exp(upper_confidence_limit) / (1 + np.exp(upper_confidence_limit))
    h_lower_confidence_limit = np.exp(lower_confidence_limit) / (1 + np.exp(lower_confidence_limit))
if LIKELIHOOD_MODEL == "poisson":
    h_estimate = np.exp(mu_posterior)
    h_upper_confidence_limit = np.exp(upper_confidence_limit)
    h_lower_confidence_limit = np.exp(lower_confidence_limit)

mu_posterior = mu_posterior.T
h_estimate = h_estimate.T
h_upper_confidence_limit = h_upper_confidence_limit.T
h_lower_confidence_limit = h_lower_confidence_limit.T

## Find observed firing rate
## Find observed firing rate
observed_mean_spikes_in_bins = zeros((N, N_plotgridpoints))
for i in range(N):
    for x in range(N_plotgridpoints):
        timesinbin = (path>bins_for_plotting[x])*(path<bins_for_plotting[x+1])
        if(sum(timesinbin)>0):
            observed_mean_spikes_in_bins[i,x] = mean( y_spikes[i, timesinbin] )
        elif i==0:
            print("No observations of X between",bins_for_plotting[x],"and",bins_for_plotting[x+1],".")
for i in range(N):
    plt.figure()
    plt.plot(x_grid_for_plotting, observed_mean_spikes_in_bins[i,:], color=plt.cm.viridis(0.1), label="Observed average")
    plt.plot(x_grid_for_plotting, h_estimate[i,:], color=plt.cm.viridis(0.5), label="Estimated expectation") 
    plt.plot(x_grid_for_plotting, h_lower_confidence_limit[i,:], "--", color=plt.cm.viridis(0.5))
    plt.plot(x_grid_for_plotting, h_upper_confidence_limit[i,:], "--", color=plt.cm.viridis(0.5))
#    plt.plot(x_grid_for_plotting, mu_posterior[i,:], color=plt.cm.viridis(0.5)) 
    plt.title("Expected and average number of spikes, neuron "+str(i)) #spikes
#    plt.title("Neuron "+str(i)+" with "+str(sum(y_spikes[i,:]))+" spikes")
    plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:])))
    plt.yticks(range(0,math.floor(max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:])))))
    plt.xlabel("x")
    plt.ylabel("Number of spikes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-tuning-"+str(i)+".png")

# Plot observed tuning for all neurons together
colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()
for i in range(N):
    plt.plot(x_grid_for_plotting, observed_mean_spikes_in_bins[i,:], color=colors[i])
#    plt.plot(x_grid_for_plotting, h_estimate[neuron[i,j],:], color=plt.cm.viridis(0.5)) 
    plt.xlabel("x")
    plt.ylabel("Average number of spikes")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-tuning-collected.png")
#plt.show()
