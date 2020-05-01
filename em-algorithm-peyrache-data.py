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

################################################
# Parameters for inference, not for generating #
################################################
T = 2000 #2000 # Max time 85504
N_iterations = 0
sigma_n = 2.5 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.99 # Learning rate by which we multiply sigma_n at every iteration

SPEEDCHECK = False
USE_OFFSET_FOR_ESTIMATE = True
NOISE_REGULARIZATION = False
N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 100 # Number of grid points for plotting f posterior only 
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "periodic" # "periodic" "nonperiodic"
sigma_f_fit = 8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.5 # Scale for the tuning curve GP that is fitted. 0.3
sigma_x = 5 # Variance of X for K_t
delta_x = 50 # Scale of X for K_t
P = 1 # Dimensions of latent variable 
GRADIENT_FLAG = False # Choose to use gradient or not

print("NBBBB!!! We're flipping the estimate after the second iteration in line 600.")
print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("Using gradient?", GRADIENT_FLAG)
print("Noise regulation:",NOISE_REGULARIZATION)
print("Initial sigma_n:", sigma_n)
print("Learning rate:", lr)
print("T:", T, "\n")
##################################
# Parameters for data generation #
##################################
downsampling_factor = 2
offset = 70400 #64460 #68170 #1000 #1751
print("Offset:", offset)
print("Downsampling factor:", downsampling_factor)
######################################
## Loading data                     ##
######################################
## 1) Load data variables
name = sys.argv[1] #'Mouse28-140313_stuff_BS0030_awakedata.mat'
mat = scipy.io.loadmat(name)
headangle = ravel(array(mat['headangle'])) # Observed head direction
cellspikes = array(mat['cellspikes']) # Observed spike time points
cellnames = array(mat['cellnames']) # Alphanumeric identifiers for cells
trackingtimes = ravel(array(mat['trackingtimes'])) # Time stamps of head direction observations
path = headangle
T_maximum = len(path)
#print("T_maximum", T_maximum)
if offset + T*downsampling_factor > T_maximum:
    sys.exit("Combination of offset, downsampling and T places the end of path outside T_maximum. Choose lower T, offset or downsampling factor.")

## 1) Remove headangles where the headangle value is NaN
# Spikes for Nan values are removed in step 2)
#print("How many NaN elements in path:", sum(np.isnan(path)))
whiches = np.isnan(path)
path = path[~whiches]

## 1.5) Make path continuous where it moves from 0 to 2pi
for t in range(1,len(path)):
    if (path[t] - path[t-1]) < - np.pi:
        path[t:] += 2*np.pi
    if (path[t] - path[t-1]) > np.pi:
        path[t:] -= 2*np.pi

## 2) Since spikes are recorded as time points, we must make a matrix with counts 0,1,2,3,4
# Here we also remove spikes that happen at NaN headangles, and then we downsample the spike matrix by summing over bins
starttime = min(trackingtimes)
tracking_interval = mean(trackingtimes[1:]-trackingtimes[:(-1)])
#print("Observation frequency for path, and binsize for initial sampling:", tracking_interval)
binsize = tracking_interval
nbins = len(trackingtimes)
#print("Number of bins for entire interval:", nbins)
print("Putting spikes in bins and making a matrix of it...")
binnedspikes = zeros((len(cellnames), nbins))
for i in range(len(cellnames)):
    spikes = ravel((cellspikes[0])[i])
    for j in range(len(spikes)):
        # note 1ms binning means that number of ms from start is the correct index
        timebin = int(floor(  (spikes[j] - starttime)/float(binsize)  ))
        if(timebin>nbins-1 or timebin<0): # check if outside bounds of the awake time
            continue
        binnedspikes[i,timebin] += 1 # add a spike to the thing

# Now remove spikes for NaN path values
binnedspikes = binnedspikes[:,~whiches]
# And downsample
binsize = downsampling_factor * tracking_interval
nbins = len(trackingtimes) // downsampling_factor
print("Bin size after downsampling: {:.2f}".format(binsize))
print("Number of bins for entire interval:", nbins)
print("Downsampling binned spikes...")
downsampled_binnedspikes = np.zeros((len(cellnames), nbins))
for i in range(len(cellnames)):
    for j in range(nbins):
        downsampled_binnedspikes[i,j] = sum(binnedspikes[i,downsampling_factor*j:downsampling_factor*(j+1)])
binnedspikes = downsampled_binnedspikes

if LIKELIHOOD_MODEL == "bernoulli":
    binnedspikes = (binnedspikes>0)*1

## 3) Select an interval of time and deal with downsampling
# We need to downsample the observed head direction when we tamper with the binsize (Here we chop off the end of the observations)
downsampled_path = np.zeros(len(path) // downsampling_factor)
for i in range(len(path) // downsampling_factor):
    downsampled_path[i] = mean(path[downsampling_factor*i:downsampling_factor*(i+1)])
path = downsampled_path
# Then do downsampled offset
downsampled_offset = offset // downsampling_factor
path = path[downsampled_offset:downsampled_offset+T]
binnedspikes = binnedspikes[:,downsampled_offset:downsampled_offset+T]

## plot head direction for the selected interval
plt.figure(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
#plt.plot(trackingtimes, path, '.', color='black', markersize=1.) # trackingtimes as x optional
#plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
plt.xlabel("Time")
plt.ylabel("x")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-headdirection.png")
plt.show()
# Plot binned spikes for the selected interval (Bernoulli style since they are binned)
bernoullispikes = (binnedspikes>0)*1
plt.figure(figsize=(10,5))
for i in range(len(cellnames)):
    plt.plot(bernoullispikes[i,:]*(i+1), '|', color='black', markersize=1.)
    plt.ylabel("neuron")
    plt.xlabel("time")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-binnedspikes.png",format="png")

## 5) Remove neurons that are not actually tuned to head direction
# On the entire range of time, these neurons are tuned to head direction
neuronsthataretunedtoheaddirection = [17,18,20,21,22,23,24,25,26,27,28,29,31,32,34,35,36,37,38,39,68]
sgood = np.zeros(len(cellnames))<1 
for i in range(len(cellnames)):
    if i not in neuronsthataretunedtoheaddirection:
        sgood[i] = False
binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]

## 6) Change names to fit the rest of the code
N = len(cellnames) #51 with cutoff at 1000 spikes
print("N:",N)
y_spikes = binnedspikes
print("mean(y_spikes)",mean(y_spikes))
print("mean(y_spikes>0)",mean(y_spikes[y_spikes>0]))
# Spike distribution evaluation
spike_count = np.ndarray.flatten(binnedspikes)
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
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-spike-histogram-log.png")

# Plot y spikes
fig, ax = plt.subplots(figsize=(8,1))
foo_mat = ax.matshow(y_spikes) #cmap=plt.cm.Blues
#fig.colorbar(foo_mat, ax=ax)
plt.title("y spikes")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-y-spikes.png")

######################
# Covariance kernels #
######################

def squared_exponential_covariance(xvector1, xvector2, sigma, delta):
    if COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = scipy.spatial.distance.cdist(xvector1, xvector2, 'sqeuclidean')
    if COVARIANCE_KERNEL_KX == "periodic":
        # First put every time point between 0 and 2pi
        xvector1 = xvector1 % (2*np.pi)
        xvector2 = xvector2 % (2*np.pi)
        # Then take care of periodicity
        distancesquared_1 = scipy.spatial.distance.cdist(xvector1, xvector2, 'sqeuclidean')
        distancesquared_2 = scipy.spatial.distance.cdist(xvector1+2*np.pi, xvector2, 'sqeuclidean')
        distancesquared_3 = scipy.spatial.distance.cdist(xvector1-2*np.pi, xvector2, 'sqeuclidean')
        min_1 = np.minimum(distancesquared_1, distancesquared_2)
        distancesquared = np.minimum(min_1, distancesquared_3)
    return sigma * exp(-distancesquared/(2*delta))

def exponential_covariance(tvector1, tvector2, sigma, delta):
    absolutedistance = scipy.spatial.distance.cdist(tvector1, tvector2, 'euclidean')
    return sigma * exp(-absolutedistance/delta)

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
    start = time.time()
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.log(1 + np.exp(F_estimate)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.exp(F_estimate))
    stop = time.time()
    if SPEEDCHECK:
        print("yf term               :", stop-start)

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
    posterior_loglikelihood = yf_term + f_prior_term + logdet_term + x_prior_term
#    if posterior_loglikelihood>0:
#        print("positive L value!!!! It should be negative.")
#        print("yf f logdet x || posterior\t",yf_term,"\t",f_prior_term,"\t",logdet_term,"\t",x_prior_term,"\t||",posterior_loglikelihood )
    #print("posterior_loglikelihood",posterior_loglikelihood)
    return - posterior_loglikelihood

def just_fprior_term(X_estimate): 
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T

    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)

    # yf_term ##########
    ####################
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.log(1 + np.exp(F_estimate)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.exp(F_estimate))

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

def without_x_prior_term(X_estimate): 
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
    start = time.time()
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.log(1 + np.exp(F_estimate)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.exp(F_estimate))
    stop = time.time()
    if SPEEDCHECK:
        print("yf term               :", stop-start)

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
    posterior_loglikelihood = yf_term + f_prior_term + logdet_term # + x_prior_term
#    if posterior_loglikelihood>0:
#        print("positive L value!!!! It should be negative.")
#        print("yf f logdet x || posterior\t",yf_term,"\t",f_prior_term,"\t",logdet_term,"\t",x_prior_term,"\t||",posterior_loglikelihood )
    #print("posterior_loglikelihood",posterior_loglikelihood)
    return - posterior_loglikelihood

def scaling(offset_for_estimate):
    scaled_estimate = X_estimate + offset_for_estimate
    return just_fprior_term(scaled_estimate)

########################
# Covariance functions #
########################
print("Making covariance matrices")

# Inducing points based on the actual range of X
x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 
print("Min and max of path:", min(path), max(path))

K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
#fig, ax = plt.subplots()
#foo_mat = ax.matshow(K_gg_plain, cmap=plt.cm.Blues)
#fig.colorbar(foo_mat, ax=ax)
#plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-kgg.png")
K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
K_gg_inverse = np.linalg.inv(K_gg)

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
#X_initial += 0.2*np.random.random(T)
#X_initial = 3 * np.ones(T)
X_initial = np.load("pretty_good_X_estimate.npy")

X_estimate = np.copy(X_initial)

# finitialize
F_initial = np.sqrt(y_spikes)
F_estimate = np.copy(F_initial)

########
# Initialize F at the values given path:
print("Setting f hat to the estimates given the true path")
temp_X_estimate = np.copy(X_estimate)
X_estimate = path
K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
K_gx_prev = K_xg_prev.T

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
true_f = np.copy(F_estimate)
## Plot F estimate
fig, ax = plt.subplots(figsize=(10,1))
foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
#fig.colorbar(foo_mat, ax=ax)
plt.title("F given path")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-F-optimal.png")
plt.clf()
plt.close()

X_estimate = temp_X_estimate
#########

## Plot initial f
fig, ax = plt.subplots(figsize=(8,1))
plt.tight_layout()
foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
#fig.colorbar(foo_mat, ax=ax)
plt.title("Initial f")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-initial-f.png")

# Speed control
SPEEDCHECK = True
x_posterior_no_la(X_estimate)
SPEEDCHECK = False
"""
# Given F, find best X value at every point without looking at the whole
# For each time bin, we try all values of X in a grid based on resolution
resolution = 20
x_trial_values = np.linspace(0,2*np.pi,resolution)
x_trial_values = x_trial_values.reshape((resolution,1))
testvalues = np.repeat(x_trial_values, T, axis=1)
L_values = np.zeros_like(testvalues) # Then we evaluate how well that position fits for every timebin, and store them
for i in range(resolution):
    for j in range(T):
        trial_X =  np.copy(X_estimate) # Make sure this is a linear something before
        trial_X[i,j] = testvalues[i,j] # L value of this 
        L_values[i,j] = without_x_prior_term(trial_X)
maxvalues  = np.amax(L_values, axis=0) # Choose best X value for every time bin (column)
for i in range(T): 
    max_position = np.where(L_values[:,i] == maxvalues[i]) # Choose best X value for every time bin.
    X_estimate[i] = testvalues[max_position]

# Then make it  continuous
"""

print("\nTest L value for different X given true F")
temp_F_estimate = np.copy(F_estimate)
F_estimate = true_f
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

collected_estimates = np.zeros((N_iterations, T))
plt.figure()
plt.title("X estimates across iterations")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.ylim((0,2*np.pi))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-collected-estimates.png")
prev_X_estimate = np.Inf
### EM algorithm: Find f given X, then X given f.
for iteration in range(N_iterations):
    print("\nIteration", iteration)
    if sigma_n > 1.9:
        sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
    print("Sigma2:", sigma_n)
    print("L value at path for this sigma:",x_posterior_no_la(path))
    print("L value at estimate for this sigma:",x_posterior_no_la(X_estimate))
    K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
    K_gg_inverse = np.linalg.inv(K_gg)

    K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx_prev = K_xg_prev.T

    # Find F estimate
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
    plt.tight_layout()
    foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
#    fig.colorbar(foo_mat, ax=ax)
    plt.title("F estimate")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-F-estimate.png")
    plt.clf()
    plt.close()

    # Attempt to explore more of the surrounding by adding noise
    if NOISE_REGULARIZATION:
        X_estimate += -0.1 + 0.2*np.random.multivariate_normal(np.zeros(T), K_t) #np.random.multivariate_normal(np.zeros(T), K_t)     #np.random.random(T)

    # Find next X estimate, that can be outside (0,2pi)
    print("Finding next X estimate...")
    print("NB! NB! We're setting the f value to the optimal F given the path.")
    F_estimate = true_f
    optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", options = {'disp':True}) #jac=x_jacobian_decoupled_la, 
    X_estimate = optimization_result.x

    # Find best offset_for_estimate
    if USE_OFFSET_FOR_ESTIMATE:
        print("\n\nFind best offset X for sigma =",sigma_n)
        initial_offset = 0
        scaling_optimization_result = optimize.minimize(scaling, initial_offset, method = "L-BFGS-B", options = {'disp':True})
        best_offset = scaling_optimization_result.x
        X_estimate = X_estimate + best_offset
        print("Best offset:", best_offset)
        # Keep it between 0 and 2Pi
        mean_of_offset_estimate = mean(X_estimate)
        X_estimate -= 2*np.pi* (mean_of_offset_estimate // (2*np.pi))
    #plt.plot(X_estimate, label='Best offset')

    plt.figure()
    plt.title("X estimates across iterations")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    collected_estimates[iteration] = np.transpose(X_estimate)
    for i in range(int(iteration+1)):
        plt.plot(collected_estimates[i], label="Estimate") #"%s" % i
    ##plt.legend(loc='upper right')
    #plt.ylim((0,2*np.pi))
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-collected-estimates.png")
    plt.clf()
    plt.close()
    np.save("X_estimate", X_estimate)
    print("Difference in X norm from last iteration:", np.linalg.norm(X_estimate - prev_X_estimate))

    plt.figure()
    plt.title("Final estimate")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    plt.plot(X_estimate, label='Estimate')
    #plt.legend(loc='upper right')
    #plt.ylim((0,2*np.pi))
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-final.png")

    if np.linalg.norm(X_estimate - prev_X_estimate) < 10**-3:
        break
    # Flipping estimate!
    if iteration == 1:
        X_estimate = 2*mean(X_estimate) - X_estimate
    prev_X_estimate = X_estimate

# Final estimate
plt.figure()
plt.title("Final estimate")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.plot(X_estimate, label='Estimate')
#plt.legend(loc='upper right')
plt.ylim((0,2*np.pi))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-final.png")
#plt.show()

###########################
# Flipped 
X_flipped = - X_estimate + 2*mean(X_estimate)

plt.figure()
plt.title("Flipped estimate")
plt.plot(X_initial, label='Initial')
plt.plot(path, color="black", label='True X')
#plt.plot(X_estimate, label='Estimate')
plt.plot(X_flipped, label='Flipped')
#plt.legend(loc='upper right')
plt.ylim((0,2*np.pi))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-flipped.png")

# Plot F estimates

#################################################
# Find posterior prediction of log tuning curve #
#################################################
bins = np.linspace(min(path)-0.000001, max(path) + 0.0000001, num=N_plotgridpoints + 1)
x_grid = 0.5*(bins[:(-1)]+bins[1:])

K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
K_gx = K_xg.T
print("Making spatial covariance matrice: Kx crossover between observations and grid")
# This one goes through inducing points
K_g_plotgrid = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)), x_grid.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)
K_plotgrid_g = K_g_plotgrid.T

fig, ax = plt.subplots()
kx_cross_mat = ax.matshow(K_g_plotgrid, cmap=plt.cm.Blues)
fig.colorbar(kx_cross_mat, ax=ax)
plt.title("Kx crossover")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-K_g_plotgrid.png")
print("Making spatial covariance matrice: Kx grid")
K_plotgrid_plotgrid = squared_exponential_covariance(x_grid.reshape((N_plotgridpoints,1)), x_grid.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)
## Keep sigma or not???
# 27.03 removing sigma from Kx grid since it will hopefully be taken care of by subtracting less (or fewer inducing points?)
#K_plotgrid_plotgrid += sigma_n*np.identity(N_plotgridpoints) # Here I am adding sigma to the diagonal because it became negative otherwise. 24.03.20
fig, ax = plt.subplots()
kxmat = ax.matshow(K_plotgrid_plotgrid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.title("Kx plotgrid")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-K_plotgrid_plotgrid.png")

Q_grid_f = np.matmul(np.matmul(K_plotgrid_g, K_gg_inverse), K_gx)
Q_f_grid = Q_grid_f.T

# Infer mean on the grid
smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
Q_ff_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
pre = np.matmul(Q_ff_plus_sigma_inverse, F_estimate.T)
mu_posterior = np.matmul(Q_grid_f, pre) # Here we have Kx crossover. Check what happens if swapped with Q = KKK
# Calculate standard deviations
sigma_posterior = K_plotgrid_plotgrid - np.matmul(Q_grid_f, np.matmul(Q_ff_plus_sigma_inverse, Q_f_grid))
fig, ax = plt.subplots()
sigma_posteriormat = ax.matshow(sigma_posterior, cmap=plt.cm.Blues)
fig.colorbar(sigma_posteriormat, ax=ax)
plt.title("sigma posterior")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-sigma_posterior.png")

###############################################
# Plot tuning curve with confidence intervals #
###############################################
standard_deviation = [np.sqrt(np.diag(sigma_posterior))]
print("posterior marginal standard deviation:\n",standard_deviation[0])
standard_deviation = np.repeat(standard_deviation, N, axis=0)
upper_confidence_limit = mu_posterior + 1.96*standard_deviation.T
lower_confidence_limit = mu_posterior - 1.96*standard_deviation.T

# if likelihood model == bernoulli
h_estimate = np.divide(np.exp(mu_posterior), (1 + np.exp(mu_posterior)))
h_upper_confidence_limit = np.exp(upper_confidence_limit) / (1 + np.exp(upper_confidence_limit))
h_lower_confidence_limit = np.exp(lower_confidence_limit) / (1 + np.exp(lower_confidence_limit))

h_estimate = h_estimate.T
h_upper_confidence_limit = h_upper_confidence_limit.T
h_lower_confidence_limit = h_lower_confidence_limit.T

## Plot neuron tuning strength in our region
# Poisson: Find average observed spike count
print("Find average observed spike count")
observed_spike_rate = zeros((len(cellnames), N_plotgridpoints))
for i in range(len(cellnames)):
    for x in range(N_plotgridpoints):
        timesinbin = (path>bins[x])*(path<bins[x+1])
        if(sum(timesinbin)>0):
            observed_spike_rate[i,x] = mean(binnedspikes[i, timesinbin] )
# Plot observed spike rates
#for n4 in range(len(cellnames)//4):
#    plt.figure(figsize=(10,8))
#    neuron = np.array([[0,1],[2,3]])
#    neuron = neuron + 4*n4
#    for i in range(2):
#        for j in range(2):
#            plt.subplot(2,2,i*2+j+1)
#            plt.plot(x_grid, observed_spike_rate[neuron[i,j],:], color=plt.cm.viridis(0.1))
#            plt.plot(x_grid, h_estimate[neuron[i,j],:], color=plt.cm.viridis(0.5)) 
#            plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_spike_rate[neuron[i,j],:])))
#            plt.xlabel("X")
#            plt.ylabel("Average number of spikes")
#            plt.title("Neuron "+str(neuron[i,j])+" with "+str(sum(binnedspikes[neuron[i,j],:]))+" spikes")
#    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-tuning"+str(n4+1)+".png")
for i in range(N):
    plt.figure()
    plt.plot(x_grid, observed_spike_rate[i,:], color=plt.cm.viridis(0.1))
    plt.plot(x_grid, h_estimate[i,:], color=plt.cm.viridis(0.5)) 
    plt.title("Neuron "+str(i)+" with "+str(sum(binnedspikes[i,:]))+" spikes")
    plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_spike_rate[i,:])))
    plt.xlabel("X")
    plt.ylabel("Average number of spikes")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-tuning-"+str(i)+".png")

plt.figure()
for i in range(N):
    plt.plot(x_grid, observed_spike_rate[i,:], color=plt.cm.viridis(0.1))
#    plt.plot(x_grid, h_estimate[neuron[i,j],:], color=plt.cm.viridis(0.5)) 
    plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_spike_rate[i,:])))
    plt.xlabel("X")
    plt.ylabel("Average number of spikes")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-tuning-"+str(i)+".png")
plt.show()
