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
numpy.random.seed(9) # 13

# This bad boi branched off from em-algorithm on 11.05.2020

################################################
# Parameters for inference, not for generating #
################################################
T = 1000 #2000 # Max time 85504
N = 100
N_iterations = 10
sigma_n = 2.5 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.95 # 0.99 # Learning rate by which we multiply sigma_n at every iteration

GRADIENT_FLAG = True # Set True to use analytic gradient
SPEEDCHECK = False
TOLERANCE = 1e-5
NOISE_REGULARIZATION = False
FLIP_AFTER_SOME_ITERATION = False
FLIP_AFTER_HOW_MANY = 6
GIVEN_TRUE_F = False
SUPREME_STARTING = False
OPTIMIZE_HYPERPARAMETERS = False
N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 40 # Number of grid points for plotting f posterior only 
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
TUNINGCURVE_DEFINITION = "bumps" # "triangles" "bumps"
UNIFORM_BUMPS = True
tuning_width = 1.2 # 0.1
baseline_f_value = -5 # -10 # -2.3 means 10 per cent chance of spiking when outside tuning area.
lambda_strength = 2 #16 ##### Set this ######
f_strength = np.log(lambda_strength) # 12 #tuning strength at bump centre
tuning_strength = f_strength - baseline_f_value
                        # This can be inspired by observed values from the head direction data if we want to. 
sigma_f_fit = 2 #8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.7 #0.5 # Scale for the tuning curve GP that is fitted. 0.3
sigma_x = 5 #5 # Variance of X for K_t
delta_x = 100 #50 # Scale of X for K_t

print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("Using gradient?", GRADIENT_FLAG)
print("Noise regulation:",NOISE_REGULARIZATION)
print("Initial sigma_n:", sigma_n)
print("Learning rate:", lr)
print("T:", T, "\n")
print("N:", N, "\n")
if FLIP_AFTER_SOME_ITERATION:
    print("NBBBB!!! We're flipping the estimate in line 600.")

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

######################################
## Data generation                  ##
######################################
bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=N_plotgridpoints + 1)
x_grid = 0.5*(bins[:(-1)]+bins[1:])

# Generative path for X:
sigma_path = 5 # Variance
delta_path = 100 # Scale 

K_t = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_path, delta_path)
K_t_inverse = np.linalg.inv(K_t)

#path = 3 + numpy.random.multivariate_normal(np.zeros(T), K_t)
#path = np.linspace(0,2*np.pi,T)
#path = np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi
path = np.pi + numpy.random.multivariate_normal(np.zeros(T), K_t)
print("Min and max of path:", min(path), max(path))
## plot path 
plt.figure(figsize=(5,2))
plt.title("Before folding")
plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
#plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
plt.xlabel("Time")
plt.ylabel("x")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robust-path.png")
# Time for some boolean masks
modulo_two_pi_values = path // (2*np.pi)
oddmodulos = (modulo_two_pi_values % 2).astype(bool)
evenmodulos = np.invert(oddmodulos)
# Even modulos: Adjust for being outside
path[evenmodulos] -= 2*np.pi*modulo_two_pi_values[evenmodulos]
# Odd modulos: Adjust for being outside and flip for continuity
path[oddmodulos] -= 2*np.pi*(modulo_two_pi_values[oddmodulos])
differences = 2*np.pi - path[oddmodulos]
path[oddmodulos] = differences
## plot path 
plt.figure(figsize=(5,2))
plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
#plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
plt.xlabel("Time")
plt.ylabel("x")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robust-path.png")
#plt.show()

## Generate spike data from a Bernoulli GLM (logistic regression) 
# True tuning curves are defined here
if UNIFORM_BUMPS:
    # Uniform positioning and width:'
    bumplocations = [(i+0.5)*2.*pi/N for i in range(N)]
    bumpwidths = tuning_width * np.ones(N)
else:
    # Random placement and width:
    bumplocations = 2*np.pi*np.random.random(N)
    bumpwidths = 0.01 + 0.5*np.random.random(N)
def bumptuningfunction(x, i): 
    x1 = x
    x2 = bumplocations[i]
    delta_x_generate = bumpwidths[i]
    if COVARIANCE_KERNEL_KX == "periodic":
        distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    elif COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = (x1-x2)**2
    return baseline_f_value + tuning_strength * exp(-distancesquared/(2*delta_x_generate))

print("Generating spikes.")
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
            true_f[i,t] = bumptuningfunction(path[t], i)
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
#        elif i==0:
#            print("No observations of X between",bins[x],"and",bins[x+1],".") 
colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
# Plot observed firing rate
plt.figure()
for i in range(N):
    plt.plot(x_grid, observed_mean_spikes_in_bins[i,:], color=colors[i])
    plt.xlabel("X")
    plt.ylabel("Average number of spikes")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-observed-spike-rates.png")

## Plot true f in time
plt.figure()
plt.xlabel("Time")
color_idx = np.linspace(0, 1, N)
plt.ylabel("True f")
for i in range(N):
    plt.plot(true_f[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-true-f-curves.png")
#plt.show()

#######
## Plot true f in the domain of X
domain_of_X = np.linspace(0,2*np.pi,T)
if TUNINGCURVE_DEFINITION == "bumps":
    f_in_domain_of_X = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            f_in_domain_of_X[i,t] = bumptuningfunction(domain_of_X[t], i)
plt.figure()
plt.xlabel("Head direction")
color_idx = np.linspace(0, 1, N)
plt.ylabel("True f")
for i in range(N):
    plt.plot(f_in_domain_of_X[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-f-in-domain-of-X.png")
#Plot true firing rate/probability in the domain of X
plt.figure()
plt.xlabel("Head direction")
color_idx = np.linspace(0, 1, N)
if LIKELIHOOD_MODEL == "bernoulli":
    plt.ylabel("True firing probability")
    plt.ylabel("Firing probability")
    for i in range(N):
        plt.plot((exp(f_in_domain_of_X[i])/(1+exp(f_in_domain_of_X[i]))), linestyle='-', color=plt.cm.viridis(color_idx[i]))
if LIKELIHOOD_MODEL == "poisson":
    plt.title("True firing rate")
    plt.ylabel("Firing rate")
    for i in range(N):
        plt.plot(exp(f_in_domain_of_X[i]), linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-firingrate-in-domain-of-X.png")
#plt.show()
#######

if LIKELIHOOD_MODEL == "bernoulli":
    ## plot true firing probability
    plt.figure()
    plt.xlabel("Time")
    color_idx = np.linspace(0, 1, N)
    plt.ylabel("True firing probability")
    plt.ylabel("Firing probability")
    for i in range(N):
        plt.plot(exp(true_f[i])/(1+exp(true_f[i])), linestyle='-', color=plt.cm.viridis(color_idx[i]))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-true-firing-probability.png")
    #plt.show()
if LIKELIHOOD_MODEL == "poisson":
    ## Plot true firing rate
    plt.figure()
    plt.xlabel("Time")
    color_idx = np.linspace(0, 1, N)
    plt.title("True firing rate")
    plt.ylabel("Firing rate")
    for i in range(N):
        plt.plot(exp(true_f[i]), linestyle='-', color=plt.cm.viridis(color_idx[i]))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-true-firing-rate.png")
    #plt.show()

## Plot true f
fig, ax = plt.subplots(figsize=(8,1))
foo_mat = ax.matshow(true_f) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("True f")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-true-f.png")
plt.clf()
plt.close()

# Plot binned spikes for selected neurons in the selected interval (Bernoulli style since they are binned)
bernoullispikes = (y_spikes>0)*1
plt.figure(figsize=(8,8))
for i in range(N):
    plt.plot(bernoullispikes[i,:]*(i+1), '|', color='black', markersize=2.)
    plt.ylabel("neuron")
    plt.xlabel("time")
plt.ylim(ymin=0.5)
plt.yticks(range(1,N+1))
#plt.yticks([9*i+1 for i in range(0,9)])
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-y_spikes.png",format="png")
#plt.show()
print("mean(y_spikes)",mean(y_spikes))
print("mean(y_spikes>0)",mean(y_spikes[y_spikes>0]))
# Spike distribution evaluation
spike_count = np.ndarray.flatten(y_spikes)
#print("This is wrong: Portion of bins with more than one spike:", sum(spike_count>1)/T)
#print("This is wrong: Portion of nonzero bins with more than one:", sum(spike_count>1) / sum(spike_count>0)) 
# Remove zero entries:
#spike_count = spike_count[spike_count>0]
plt.figure()
plt.hist(spike_count, bins=np.arange(0,int(max(spike_count))+1)-0.5, log=True, color=plt.cm.viridis(0.3))
plt.ylabel("Number of bins")
plt.xlabel("Spike count")
plt.title("Spike histogram")
plt.xticks(range(0,int(max(spike_count)),1))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-spike-histogram-log.png")

# Plot y spikes
fig, ax = plt.subplots() # figsize=(8,1)
foo_mat = ax.matshow(y_spikes) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("y spikes")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-y-spikes.png")

#########################
## Likelihood functions #
#########################

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize. Equation (4.17)++
def f_loglikelihood_bernoulli(f_i): # Psi
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.log(1+np.exp(f_i))) # Corrected 16.03 from sum( np.multiply(y_i, (f_i - np.log(1+np.exp(f_i)))) + np.multiply((1-y_i), np.log(1- np.divide(np.exp(f_i), 1 + np.exp(f_i)))))
    priorterm_1 = -0.5*sigma_n**-2 * np.dot(f_i.T, f_i)
    fT_k = np.dot(f_i, K_xg_prev)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = 0.5*sigma_n**-2 * np.dot(np.dot(fT_k, smallinverse), fT_k.T)
    return - (likelihoodterm + priorterm_1 + priorterm_2)
def f_jacobian_bernoulli(f_i):
    yf_term = y_i - np.divide(np.exp(f_i), 1 + np.exp(f_i))
    priorterm_1 = -sigma_n**-2 * f_i
    kTf = np.dot(K_xg_prev.T, f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, kTf))
    f_derivative = yf_term + priorterm_1 + priorterm_2
    return - f_derivative
def f_hessian_bernoulli(f_i):
    e_tilde = np.divide(np.exp(f_i), (1 + np.exp(f_i))**2)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    f_hessian = - np.diag(e_tilde) - sigma_n**-2 * np.identity(T) + sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, K_xg_prev.T))
    return - f_hessian

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize.
def f_loglikelihood_poisson(f_i):
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.exp(f_i)) 
    priorterm_1 = -0.5*sigma_n**-2 * np.dot(f_i.T, f_i)
    fT_k = np.dot(f_i, K_xg_prev)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = 0.5*sigma_n**-2 * np.dot(np.dot(fT_k, smallinverse), fT_k.T)
    return - (likelihoodterm + priorterm_1 + priorterm_2)

def f_jacobian_poisson(f_i):
    yf_term = y_i - np.exp(f_i)
    priorterm_1 = -sigma_n**-2 * f_i
    kTf = np.dot(K_xg_prev.T, f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, kTf))
    f_derivative = yf_term + priorterm_1 + priorterm_2
    return - f_derivative
def f_hessian_poisson(f_i):
    e_poiss = np.exp(f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    f_hessian = - np.diag(e_poiss) - sigma_n**-2*np.identity(T) + sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, K_xg_prev.T))
    return - f_hessian

# L function
def x_posterior_no_la(X_estimate): 
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

    #print("f_prior_term",f_prior_term)
    #print("logdet_term",logdet_term)
    #print("x_prior_term",x_prior_term)
    posterior_loglikelihood = logdet_term + f_prior_term + x_prior_term #+ yf_term
#    if posterior_loglikelihood>0:
#        print("positive L value!!!! It should be negative.")
#        print("yf f logdet x || posterior\t",yf_term,"\t",f_prior_term,"\t",logdet_term,"\t",x_prior_term,"\t||",posterior_loglikelihood )
    #print("posterior_loglikelihood",posterior_loglikelihood)
    return - posterior_loglikelihood

# Gradient of L 
def x_jacobian_no_la(X_estimate):
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
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)
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
    f_prior_gradient = sigma_n**-2 / 2 * fMfsum

    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)

    ####################
    # x prior term #####
    ####################
    start = time.time()
    x_prior_gradient = - np.dot(X_estimate.T, K_t_inverse)
    stop = time.time()
    if SPEEDCHECK:
        print("X prior term          :", stop-start)
    ####################

    #print("logdet_gradient\n", logdet_gradient)
    #print("f_prior_gradient\n",f_prior_gradient) 
    #print("x_prior_gradient\n", x_prior_gradient)

    x_gradient = logdet_gradient + f_prior_gradient + x_prior_gradient 
    return - x_gradient

########################
# Covariance matrices  #
########################
print("Making covariance matrices")

# Inducing points based on the actual range of X
x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 
print("Min and max of path:", min(path), max(path))

K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
#fig, ax = plt.subplots()
#foo_mat = ax.matshow(K_gg_plain) #cmap=plt.cm.Blues
#fig.colorbar(foo_mat, ax=ax)
#plt.title("Kgg plain")
#plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-kgg-plain.png")
K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)

## Plot Kgg
fig, ax = plt.subplots()
foo_mat = ax.matshow(K_gg) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("Kgg")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-kgg-plain.png")
#plt.show()
#plt.clf()
#plt.close()

K_t = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)

######################
# Initialize X and F #
######################
# xinitialize
#X_initial = 2 * np.ones(T)
#X_initial = 5 * np.ones(T) - 4*np.linspace(0,T,T)/T
#X_initial[0:100] = 5 - 3*np.linspace(0,100,100)/100
#X_initial[1200:1500] = 2 + 3*np.linspace(0,300,300)/300
#X_initial[1500:2000] = 5
#X_initial = np.load("X_estimate_supreme.npy")
#X_initial = np.load("X_estimate.npy")
#X_initial = 2 + np.linspace(0,T,T)/T
X_initial = 1.5 * np.ones(T)
X_initial += 0.2*np.random.random(T)

X_estimate = np.copy(X_initial)
if SUPREME_STARTING:
    X_estimate = np.load("X_estimate_supreme.npy")

# finitialize
F_initial = np.sqrt(y_spikes) - np.amax(np.sqrt(y_spikes))/2 #np.sqrt(y_spikes) - 2 
if SUPREME_STARTING:
    F_initial = np.load("F_estimate_supreme.npy")
F_estimate = np.copy(F_initial)
if GIVEN_TRUE_F:
    F_estimate = true_f

#print("Testing gradient...")
#X_estimate = path + 0.3 
#F_estimate = true_f
#print("Gradient difference using check_grad:",scipy.optimize.check_grad(x_posterior_no_la, x_jacobian_no_la, path))
#plt.show()
#SPEEDCHECK = True
#x_jacobian_no_la(X_estimate)
#SPEEDCHECK = False

"""
#optim_gradient = optimization_result.jac
print("Epsilon:", np.sqrt(np.finfo(float).eps))
optim_gradient1 = scipy.optimize.approx_fprime(X_estimate, x_posterior_no_la, 1*np.sqrt(np.finfo(float).eps))
optim_gradient2 = scipy.optimize.approx_fprime(X_estimate, x_posterior_no_la, 1e-4)
optim_gradient3 = scipy.optimize.approx_fprime(X_estimate, x_posterior_no_la, 1e-2)
optim_gradient4 = scipy.optimize.approx_fprime(X_estimate, x_posterior_no_la, 1e-2)
calculated_gradient = x_jacobian_no_la(X_estimate)
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
plt.title("Gradient, all terms")
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
"""

## Plot initial f
fig, ax = plt.subplots(figsize=(8,2))
foo_mat = ax.matshow(F_initial) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("Initial f")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-initial-f.png")

# Speed control
SPEEDCHECK = True
x_posterior_no_la(X_estimate)
SPEEDCHECK = False

"""
print("\nTest L value for different X given true F")
temp_F_estimate = np.copy(F_estimate)
if GIVEN_TRUE_F:
    F_estimate = np.copy(true_f)
tempsigma = sigma_n
for sigma in [3.0, 2.5, 2.0, 1.5, 2.0, 1.5, 1.0, 0.5, 0.1]:
    sigma_n = sigma
    print("Sigma", sigma_n)
    print("path\n", x_posterior_no_la(path))
    print("path + 0.1*np.random.random(T)\n",x_posterior_no_la(path + 0.1*np.random.random(T)))
    print("path + 0.2*np.random.random(T)\n",x_posterior_no_la(path + 0.2*np.random.random(T)))
    print("path + 0.3*np.random.random(T)\n",x_posterior_no_la(path + 0.3*np.random.random(T)))
    print("path - 0.3\n",x_posterior_no_la(path - 0.3))
    print("path - 0.2\n",x_posterior_no_la(path - 0.2))
    print("path - 0.1\n",x_posterior_no_la(path - 0.1))
    print("path + 0.1\n",x_posterior_no_la(path + 0.1))
    print("path + 0.2\n",x_posterior_no_la(path + 0.2))
    print("path + 0.3\n",x_posterior_no_la(path + 0.3))
    print("Random start\n",x_posterior_no_la(2*np.pi*np.random.random(T)), "\n")
sigma_n = tempsigma
F_estimate = temp_F_estimate
"""

collected_estimates = np.zeros((N_iterations, T))
plt.figure()
plt.title("X estimates across iterations")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-EM-collected-estimates.png")
plt.clf()
plt.close()
prev_X_estimate = np.Inf
### EM algorithm: Find f given X, then X given f.
for iteration in range(N_iterations):
    print("\nIteration", iteration)
    if iteration > 0:
        sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
    print("Sigma2:", sigma_n)
    print("L value at path for this sigma:",x_posterior_no_la(path))
    print("L value at estimate for this sigma:",x_posterior_no_la(X_estimate))
    K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)

    K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx_prev = K_xg_prev.T

    # Find F estimate only if we're not at the first iteration
    if iteration > 0:
        print("Finding f hat...")
        if LIKELIHOOD_MODEL == "bernoulli":
            for i in range(N):
                y_i = y_spikes[i]
                optimization_result = optimize.minimize(f_loglikelihood_bernoulli, F_estimate[i], jac=f_jacobian_bernoulli, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
                F_estimate[i] = optimization_result.x
        elif LIKELIHOOD_MODEL == "poisson":
            for i in range(N):
                y_i = y_spikes[i]
                optimization_result = optimize.minimize(f_loglikelihood_poisson, F_estimate[i], jac=f_jacobian_poisson, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
                F_estimate[i] = optimization_result.x 
        ## Plot F estimate
        fig, ax = plt.subplots(figsize=(8,1))
        foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
        fig.colorbar(foo_mat, ax=ax)
        plt.title("F estimate")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-F-estimate.png")
        plt.clf()
        plt.close()
    else: 
        print("First iteration: Skipping F inference.")

    # Find next X estimate, that can be outside (0,2pi)
    print("Finding next X estimate...")
    if GIVEN_TRUE_F: 
        print("NB! NB! We're setting the f value to the optimal F given the path.")
        F_estimate = np.copy(true_f)
    if GRADIENT_FLAG: 
        optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':True})
    else:
        optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", options = {'disp':True})
    X_estimate = optimization_result.x

    if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
        # Flipping estimate after iteration 1 has been plotted
        X_estimate = 2*mean(X_estimate) - X_estimate

    plt.figure()
    plt.title("X estimates across iterations")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    collected_estimates[iteration] = np.transpose(X_estimate)
    for i in range(int(iteration+1)):
        plt.plot(collected_estimates[i], label="Estimate") #"%s" % i
    ##plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-EM-collected-estimates.png")
    plt.clf()
    plt.close()

    plt.figure()
    plt.title("X Estimate") # as we go
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    plt.plot(X_estimate, label='Estimate')
    plt.legend()
    #plt.ylim((0,2*np.pi))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-EM-as-we-go.png")

    np.save("X_estimate", X_estimate)
    print("Difference in X norm from last iteration:", np.linalg.norm(X_estimate - prev_X_estimate))
    if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
        break
    prev_X_estimate = X_estimate
## Flipped 
#X_flipped = - X_estimate + 2*mean(X_estimate)
## Rootmeansquarederror for X
#X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
#X_flipped_rmse = np.sqrt(sum((X_flipped-path)**2) / T)
#print("First convergence")
#print("RMSE for X:", X_rmse)
#print("RMSE Flipped X:", X_flipped_rmse)
#print("\n")
#
###### Check if flipped and maybe iterate again with flipped estimate
#if X_flipped_rmse < X_rmse:
#    X_estimate = X_flipped
#    prev_X_estimate = np.Inf
#    for iteration in range(N_iterations):
#        if iteration > 0:
#            sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
#        K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
#        K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
#        K_gx_prev = K_xg_prev.T
#        # Find F estimate only if we're not at the first iteration
#        if iteration > 0:
#            if LIKELIHOOD_MODEL == "bernoulli":
#                for i in range(N):
#                    y_i = y_spikes[i]
#                    optimization_result = optimize.minimize(f_loglikelihood_bernoulli, F_estimate[i], jac=f_jacobian_bernoulli, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
#                    F_estimate[i] = optimization_result.x
#            elif LIKELIHOOD_MODEL == "poisson":
#                for i in range(N):
#                    y_i = y_spikes[i]
#                    optimization_result = optimize.minimize(f_loglikelihood_poisson, F_estimate[i], jac=f_jacobian_poisson, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
#                    F_estimate[i] = optimization_result.x 
#        # Find next X estimate, that can be outside (0,2pi)
#        if GIVEN_TRUE_F: 
#            print("NB! NB! We're setting the f value to the optimal F given the path.")
#            F_estimate = np.copy(true_f)
#        if GRADIENT_FLAG: 
#            optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
#        else:
#            optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", options = {'disp':False})
#        X_estimate = optimization_result.x
#        if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
#            # Flipping estimate after iteration 1 has been plotted
#            X_estimate = 2*mean(X_estimate) - X_estimate
#        plt.figure()
#        plt.title("X estimates across iterations")
#        plt.plot(path, color="black", label='True X')
#        plt.plot(X_initial, label='Initial')
#        collected_estimates[iteration] = np.transpose(X_estimate)
#        for i in range(int(iteration+1)):
#            plt.plot(collected_estimates[i], label="Estimate") #"%s" % i
#        ##plt.legend(loc='upper right')
#        plt.tight_layout()
#        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-EM-collected-estimates.png")
#        plt.clf()
#        plt.close()
#
#        plt.figure()
#        plt.title("X Estimate") # as we go
#        plt.plot(path, color="black", label='True X')
#        plt.plot(X_initial, label='Initial')
#        plt.plot(X_estimate, label='Estimate')
#        plt.legend()
#        #plt.ylim((0,2*np.pi))
#        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-EM-as-we-go.png")
#
#        if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
#            break
#        prev_X_estimate = X_estimate

# Final estimate
plt.figure()
plt.title("Head direction")
plt.plot(path, color="black", label='Observed')
plt.plot(X_initial, label='Initial')
plt.plot(X_estimate, label='Estimate')
plt.ylabel("X")
plt.xlabel("Timebin")
plt.legend() #loc='upper right'
plt.ylim((0,2*np.pi))
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-EM-final.png")
#plt.show()
###########################
# Flipped 
X_flipped = - X_estimate + 2*mean(X_estimate)

plt.figure()
plt.title("Flipped estimate")
#plt.plot(X_initial, label='Initial')
plt.plot(path, color="black", label='True X')
#plt.plot(X_estimate, label='Estimate')
plt.plot(X_flipped, label='Flipped')
#plt.legend(loc='upper right')
plt.ylim((0,2*np.pi))
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-EM-flipped.png")

# Save F estimates
np.save("F_estimate_em",F_estimate)

rootmeansquarederror = np.sqrt(sum((X_estimate-path)**2) / T)
flippedrootmeansquarederror = np.sqrt(sum((X_flipped-path)**2) / T)
print("\n")
print("Baseline expected spikes:", np.exp(baseline_f_value))
print("Expected no. of spikes at peak:", lambda_strength)
print("f value at peak:", baseline_f_value + tuning_strength)
print("RMSE:", rootmeansquarederror)
print("RMSE Flipped X:", flippedrootmeansquarederror)
SStot = sum((path - mean(path))**2)
SSdev = sum((X_estimate-path)**2)
Rsquared = 1 - SSdev / SStot
#print("R squared value of X estimate:", Rsquared, "\n")

#exit()


###########################################
# Find point estimates of hyperparameters #
###########################################
def hyperparam_loglikelihood(hyperparameters):
    sigma_x = hyperparameters[0]
    delta_x = hyperparameters[1]
    sigma_f_fit = hyperparameters[2]
    delta_f_fit = hyperparameters[3]
    #sigma_n = hyperparameters[4]
    # Create covariance matrices
    K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
    K_t = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_x, delta_x)
    K_t_inverse = np.linalg.inv(K_t)
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T

    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)

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
    xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)

    #print("f_prior_term",f_prior_term)
    #print("logdet_term",logdet_term)
    #print("x_prior_term",x_prior_term)
    posterior_loglikelihood = f_prior_term + logdet_term + x_prior_term
    #print("posterior_loglikelihood",posterior_loglikelihood)
    return - posterior_loglikelihood

if OPTIMIZE_HYPERPARAMETERS: 
    sigma_n = 1.5
    # length and sigma for X, length and sigma for f, noise param
    #initial_hyperparameters = [sigma_x, delta_x, sigma_f_fit, delta_f_fit, sigma_n] 
    initial_hyperparameters = [sigma_x, delta_x, sigma_f_fit, delta_f_fit] # sigma_n 
    bnds = ((0, None), (0, None), (0, None), (0, None))
    hyper_optim_result = optimize.minimize(hyperparam_loglikelihood, initial_hyperparameters, method = "L-BFGS-B", bounds=bnds, options = {'disp':True})
    optimized_hyperparameters = hyper_optim_result.x
    print("sigma_x:", optimized_hyperparameters[0])
    print("delta_x:", optimized_hyperparameters[1])
    print("sigma_f_fit:", optimized_hyperparameters[2])
    print("delta_f_fit:", optimized_hyperparameters[3])
    #print("sigma_n:", optimized_hyperparameters[4])

#################################################
# Find posterior prediction of log tuning curve #
#################################################
f_values_observed = F_estimate

def exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def gaussian_periodic_covariance(x1,x2, sigma, delta):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return sigma * exp(-distancesquared/(2*delta))

def gaussian_NONPERIODIC_covariance(x1,x2, sigma, delta):
    distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

# Inducing points 
x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 

K_fu = np.zeros((T,N_inducing_points))
for x1 in range(T):
    for x2 in range(N_inducing_points):
        K_fu[x1,x2] = gaussian_periodic_covariance(path[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
K_uf = K_fu.T

# To be absolutely certain, we make Kx inducing again: 
K_uu = np.zeros((N_inducing_points,N_inducing_points))
for x1 in range(N_inducing_points):
    for x2 in range(N_inducing_points):
        K_uu[x1,x2] = gaussian_periodic_covariance(x_grid_induce[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
#K_uu = K_uu + sigma_n/np.sqrt(T) *np.identity(N_inducing_points)
K_uu += 0.05*np.identity(N_inducing_points)
K_uu_inverse = np.linalg.inv(K_uu)

print("Making spatial covariance matrice: Kx crossover beween observations and grid")
# Goes through inducing points
K_u_grid = np.zeros((N_inducing_points,N_plotgridpoints))
for x1 in range(N_inducing_points):
    for x2 in range(N_plotgridpoints):
        K_u_grid[x1,x2] = gaussian_periodic_covariance(x_grid_induce[x1],x_grid[x2], sigma_f_fit, delta_f_fit)
K_grid_u = K_u_grid.T

fig, ax = plt.subplots()
kx_cross_mat = ax.matshow(K_u_grid, cmap=plt.cm.Blues)
fig.colorbar(kx_cross_mat, ax=ax)
plt.title("Kx crossover")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-K_u_grid.png")
print("Making spatial covariance matrice: Kx grid")
K_grid_grid = np.zeros((N_plotgridpoints,N_plotgridpoints))
for x1 in range(N_plotgridpoints):
    for x2 in range(N_plotgridpoints):
        K_grid_grid[x1,x2] = gaussian_periodic_covariance(x_grid[x1],x_grid[x2], sigma_f_fit, delta_f_fit)
# 27.03 removing sigma from Kx grid since it will hopefully be taken care of by subtracting less (or fewer inducing points?)
#K_grid_grid += sigma_n*np.identity(N_plotgridpoints) # Here I am adding sigma to the diagonal because it became negative otherwise. 24.03.20
fig, ax = plt.subplots()
kxmat = ax.matshow(K_grid_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.title("Kx grid")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-K_grid_grid.png")

Q_grid_f = np.matmul(np.matmul(K_grid_u, K_uu_inverse), K_uf)
Q_f_grid = Q_grid_f.T

# Infer mean on the grid
smallinverse = np.linalg.inv(K_uu*sigma_n**2 + np.matmul(K_uf, K_fu))
Q_ff_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_fu, smallinverse), K_uf)
pre = np.matmul(Q_ff_plus_sigma_inverse, f_values_observed.T)
mu_posterior = np.matmul(Q_grid_f, pre) # Here we have Kx crossover. Check what happens if swapped with Q = KKK
# Calculate standard deviations
sigma_posterior = K_grid_grid - np.matmul(Q_grid_f, np.matmul(Q_ff_plus_sigma_inverse, Q_f_grid))
fig, ax = plt.subplots()
sigma_posteriormat = ax.matshow(sigma_posterior, cmap=plt.cm.Blues)
fig.colorbar(sigma_posteriormat, ax=ax)
plt.title("sigma posterior")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-sigma_posterior.png")

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
observed_mean_spikes_in_bins = zeros((N, N_plotgridpoints))
for i in range(N):
    for x in range(N_plotgridpoints):
        timesinbin = (path>bins[x])*(path<bins[x+1])
        if(sum(timesinbin)>0):
            observed_mean_spikes_in_bins[i,x] = mean( y_spikes[i, timesinbin] )
        elif i==0:
            print("No observations of X between",bins[x],"and",bins[x+1],".")
for i in range(N):
    plt.figure()
    plt.plot(x_grid, observed_mean_spikes_in_bins[i,:], color=plt.cm.viridis(0.1), label="Observed")
    plt.plot(x_grid, h_estimate[i,:], color=plt.cm.viridis(0.5), label="Estimated") 
#    plt.plot(x_grid, mu_posterior[i,:], color=plt.cm.viridis(0.5)) 
    plt.title("Average number of activities, neuron "+str(i)) #spikes
#    plt.title("Neuron "+str(i)+" with "+str(sum(y_spikes[i,:]))+" spikes")
    plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_mean_spikes_in_bins[i,:])))
    plt.xlabel("X")
#    plt.ylabel("Number of spikes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-tuning-"+str(i)+".png")

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()
for i in range(N):
    plt.plot(x_grid, observed_mean_spikes_in_bins[i,:], color=colors[i])
#    plt.plot(x_grid, h_estimate[neuron[i,j],:], color=plt.cm.viridis(0.5)) 
    plt.xlabel("X")
    plt.ylabel("Average number of spikes")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-tuning-collected.png")
#plt.show()
