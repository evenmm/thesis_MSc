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

## One lambda, one path with seed 12
## For different inital seeds and either wide random or offset random initials: 
## Infer final estimate and save it with rmse and L value

################################################
# Parameters for inference, not for generating #
################################################
T = 1000 #2000 # Max time 85504
N = 100
N_iterations = 30

global_initial_sigma_n = 2.5 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.95 # 0.99 # Learning rate by which we multiply sigma_n at every iteration

GRADIENT_FLAG = True # Set True to use analytic gradient
SPEEDCHECK = False
TOLERANCE = 1e-3
NOISE_REGULARIZATION = False
FLIP_AFTER_SOME_ITERATION = False
FLIP_AFTER_HOW_MANY = 10
GIVEN_TRUE_F = False
OPTIMIZE_HYPERPARAMETERS = False
PLOTTING = True
N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 40 # Number of grid points for plotting f posterior only 
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
TUNINGCURVE_DEFINITION = "bumps" # "triangles" "bumps"
UNIFORM_BUMPS = True
tuning_width = 1.2 # 0.1
baseline_f_value = -10 # -2.3 means 10 per cent chance of spiking when outside tuning area.
lambda_strength_array = [8]
seeds = [12] 
NUMBER_OF_SEEDS = len(seeds)
print("Number of seeds we average over:", NUMBER_OF_SEEDS)
sigma_f_fit = 2 #8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.7 #0.5 # Scale for the tuning curve GP that is fitted. 0.3
sigma_x = 5 #1 #5 # Variance of X for K_t
delta_x = 100 #50 # Scale of X for K_t

print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("Using gradient?", GRADIENT_FLAG)
print("Noise regulation:",NOISE_REGULARIZATION)
print("Initial sigma_n:", global_initial_sigma_n)
print("Learning rate:", lr)
print("T:", T)
print("N:", N)
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

######################################
## Data generation                  ##
######################################
bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=N_plotgridpoints + 1)
x_grid = 0.5*(bins[:(-1)]+bins[1:])
# Generative parameters for X path:
sigma_path = 5 # Variance
delta_path = 100 # Scale 
K_t_generate = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_path, delta_path)
if UNIFORM_BUMPS:
    # Uniform positioning and width:'
    bumplocations = [(i+0.5)*2.*pi/N for i in range(N)]
    bumpwidths = tuning_width * np.ones(N)
else:
    # Random placement and width:
    bumplocations = 2*np.pi*np.random.random(N)
    bumpwidths = 0.01 + 0.5*np.random.random(N)
def bumptuningfunction(x, i, tuning_strength): 
    x1 = x
    x2 = bumplocations[i]
    delta_x_generate = bumpwidths[i]
    if COVARIANCE_KERNEL_KX == "periodic":
        distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    elif COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = (x1-x2)**2
    return baseline_f_value + tuning_strength * exp(-distancesquared/(2*delta_x_generate))

######################################
## RMSE function                  ##
######################################
def find_rmse_for_this_lambda_this_seed(seedindex):
    tuning_strength = np.log(lambda_strength) - baseline_f_value # f_strength - baseline_f_value # 12 #tuning strength at bump centre
    np.random.seed(seeds[0])
    path = np.pi + numpy.random.multivariate_normal(np.zeros(T), K_t_generate)
    # Use boolean masks to keep X within (0, 2pi)
    modulo_two_pi_values = path // (2*np.pi)
    oddmodulos = (modulo_two_pi_values % 2).astype(bool)
    evenmodulos = np.invert(oddmodulos)
    # Even modulos: Adjust for being outside
    path[evenmodulos] -= 2*np.pi*modulo_two_pi_values[evenmodulos]
    # Odd modulos: Adjust for being outside and flip for continuity
    path[oddmodulos] -= 2*np.pi*(modulo_two_pi_values[oddmodulos])
    differences = 2*np.pi - path[oddmodulos]
    path[oddmodulos] = differences
    if PLOTTING:
        ## plot path 
        plt.figure(figsize=(5,2))
        plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
        #plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-T-" + str(T) + "-path.png")
    ## Generate spike data. True tuning curves are defined here
    if TUNINGCURVE_DEFINITION == "triangles":
        tuningwidth = 1 # width of tuning (in radians)
        biasterm = -2 # Average H outside tuningwidth -4
        tuningcovariatestrength = np.linspace(0.5*tuningwidth,10.*tuningwidth, N) # H value at centre of tuningwidth 6*tuningwidth
        neuronpeak = [(i+0.5)*2.*pi/N for i in range(N)]
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
                true_f[i,t] = bumptuningfunction(path[t], i, tuning_strength)
                if LIKELIHOOD_MODEL == "bernoulli":
                    spike_probability = exp(true_f[i,t])/(1.+exp(true_f[i,t]))
                    y_spikes[i,t] = 1.0*(rand()<spike_probability)
                elif LIKELIHOOD_MODEL == "poisson":
                    spike_rate = exp(true_f[i,t])
                    y_spikes[i,t] = np.random.poisson(spike_rate)
    ## Find observed firing rate
    observed_mean_spikes_in_bins = zeros((N, N_plotgridpoints))
    for i in range(N):
        for x in range(N_plotgridpoints):
            timesinbin = (path>bins[x])*(path<bins[x+1])
            if(sum(timesinbin)>0):
                observed_mean_spikes_in_bins[i,x] = mean( y_spikes[i, timesinbin] )
    ######################
    # Initialize X and F #
    ######################
    np.random.seed(seedindex)
    print("Initializing X using seed", starting_seed_array[seedindex])
    X_initial = 1 * np.ones(T)
    X_estimate = np.copy(X_initial)
    #epsilon_fstart = 1e-3
    #F_initial = np.log(y_spikes + epsilon_fstart)
    F_initial = np.sqrt(y_spikes) - np.amax(np.sqrt(y_spikes))/2 #np.sqrt(y_spikes) - 2
    F_estimate = np.copy(F_initial)
    if GIVEN_TRUE_F:
        F_estimate = true_f
    ###############################
    # Covariance matrix Kgg_plain #
    ###############################
    # Inducing points based on the actual range of X    
    x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 
    #x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) 
    K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)    
    #############################
    # Iterate with EM algorithm #
    #############################
    if PLOTTING:
        if T == 1000:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        plt.title("Initial estimate") # as we go
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.plot(path, color="black", label='True X')
        plt.plot(X_initial, label='Initial')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-T-" + str(T) + "-lambda-" + str(lambda_strength) + "-seed-" + str(starting_seed_array[seedindex]) + ".png")
    if PLOTTING:
        if T == 1000:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        plt.title("X estimate") # as we go
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.plot(path, color="black", label='True X')
        plt.plot(X_initial, label='Initial')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-T-" + str(T) + "-lambda-" + str(lambda_strength) + "-seed-" + str(starting_seed_array[seedindex]) + ".png")
    prev_X_estimate = np.Inf
    sigma_n = np.copy(global_initial_sigma_n)
    for iteration in range(N_iterations):
        if iteration > 0:
            sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
        #x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) 
        K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
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
        if GRADIENT_FLAG: 
            optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
        else:
            optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
        X_estimate = optimization_result.x
        if PLOTTING:
            plt.plot(X_estimate, label='Estimate')
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-T-" + str(T) + "-lambda-" + str(lambda_strength) + "-seed-" + str(starting_seed_array[seedindex]) + ".png")
        if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
            # Flipping estimate after iteration 1 has been plotted
            X_estimate = 2*mean(X_estimate) - X_estimate
        if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
            break
        prev_X_estimate = X_estimate
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
        sigma_n = np.copy(global_initial_sigma_n)
        X_initial_2 = np.copy(X_flipped)
        X_estimate = np.copy(X_flipped)
        F_estimate = np.copy(F_initial)
        if PLOTTING:
            if T == 1000:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            plt.title("X estimate")
            plt.xlabel("Time")
            plt.ylabel("x")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_initial_2, label='Initial')
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-T-" + str(T) + "-lambda-" + str(lambda_strength) + "-seed-" + str(starting_seed_array[seedindex]) + "-flipped.png")
        prev_X_estimate = np.Inf
        for iteration in range(N_iterations):
            if iteration > 0:
                sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
            K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
            K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
            # Here we want to find a new F estimate regardless
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
            if GRADIENT_FLAG: 
                optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
            else:
                optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
            X_estimate = optimization_result.x
            if PLOTTING:
                plt.plot(X_estimate, label='Estimate') #after flip
                plt.tight_layout()
                plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-T-" + str(T) + "-lambda-" + str(lambda_strength) + "-seed-" + str(starting_seed_array[seedindex]) + "-flipped.png")
            if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
                # Flipping estimate after iteration 1 has been plotted
                X_estimate = 2*mean(X_estimate) - X_estimate
            if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
                break
            prev_X_estimate = X_estimate
        # Rootmeansquarederror for X
        X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
    if PLOTTING:
        if T == 1000:
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
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-T-" + str(T) + "-lambda-" + str(lambda_strength) + "-seed-" + str(starting_seed_array[seedindex]) + "-final.png")
    print("Seed", starting_seed_array[seedindex], "finished. RMSE for X:", X_rmse)
    L_value = x_posterior_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
    F_estimate = true_f
    L_value_true_f = x_posterior_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
    return [X_initial, X_estimate, X_rmse, L_value, L_value_true_f, seedindex]

if __name__ == "__main__": 
    starting_seed_array = [0] #range(7)
    # We gather initials, finals, rmse values and L values
    initial_array = np.zeros((N,T))
    final_array = np.zeros((N,T))
    rmse_values = np.zeros(len(starting_seed_array))
    L_values = np.zeros(len(starting_seed_array))
    L_values_true_f = np.zeros(len(starting_seed_array))
    seed_index_array = np.zeros(len(starting_seed_array))
    for lambda_index in range(len(lambda_strength_array)):
        global lambda_strength
        lambda_strength = lambda_strength_array[lambda_index]

        # Pool computing
        starttime = time.time()
        myPool = Pool(processes=8) #len(starting_seed_array))
        result_array = myPool.map(find_rmse_for_this_lambda_this_seed, [i for i in range(len(starting_seed_array))])
        myPool.close()
        endtime = time.time()
        print("Time use:", endtime - starttime)

        # Unpack results
        for i in range(len(starting_seed_array)):
            initial_array[i] = result_array[i][0]
            final_array[i] = result_array[i][1]
            rmse_values[i] = result_array[i][2]
            L_values[i] = result_array[i][3]
            L_values_true_f[i] = result_array[i][4]
            seed_index_array[i] = result_array[i][5]
            
        print("rmse:\n", rmse_values)
        print("L values with corresponding F estimate\n", L_values)
        print("L values with true F\n", L_values_true_f)

        np.random.seed(seeds[0])
        path = np.pi + numpy.random.multivariate_normal(np.zeros(T), K_t_generate)
        if T == 1000:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.title("Initial estimates") 
        plt.plot(path, color="black", label='True X')
        for i in range(len(starting_seed_array)):
            plt.plot(initial_array[i])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-initial.png")
        if T == 1000:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        plt.xlabel("Time")
        plt.ylabel("x")
        plt.title("Final estimates") 
        plt.plot(path, color="black", label='True X')
        for i in range(len(starting_seed_array)):
            plt.plot(final_array[i])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-ii-final.png")

