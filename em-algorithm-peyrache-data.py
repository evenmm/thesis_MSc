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
T = 200
N = 100
N_iterations = 5
sigma_n = 2.5 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.9 # Learning rate by which we multiply sigma_n at every iteration

N_inducing_points = 25 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 100 # Number of grid points for plotting f posterior only 
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "periodic" # "periodic" "nonperiodic"
sigma_f_fit = 8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 1 # Scale for the tuning curve GP that is fitted. 0.3
sigma_x = 5 # Variance of X for K_t
delta_x = 10 # Scale of X for K_t
P = 1 # Dimensions of latent variable 
GRADIENT_FLAG = False # Choose to use gradient or not

print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("Using gradient?", GRADIENT_FLAG)

##################################
# Parameters for data generation #
##################################
downsampling_factor = 1
offset = 1650 #1750
thresholdforneuronstokeep = 1000 # number of spikes to be considered useful

######################################
## Loading data                     ##
######################################
name = sys.argv[1] #'Mouse28-140313_stuff_BS0030_awakedata.mat'

mat = scipy.io.loadmat(name)
headangle = ravel(array(mat['headangle'])) # Observed head direction
cellspikes = array(mat['cellspikes']) # Observed spike time points
cellnames = array(mat['cellnames']) # Alphanumeric identifiers for cells
trackingtimes = ravel(array(mat['trackingtimes'])) # Time stamps of path observations

## define bin size and make matrix of spikes
startt = min(trackingtimes)
tracking_interval = mean(trackingtimes[1:]-trackingtimes[:(-1)])
print("Interval between head direction observations:", tracking_interval)
binsize = downsampling_factor * tracking_interval
print("Bin size for spike counts:", binsize)
print("Make sure that path and spikes are still aligned!!!")
# We need to downsample the observed head direction as well when we tamper with the binsize
#downsampled = np.zeros(len(headangle) // downsampling_factor)
#for i in range(len(headangle) // downsampling_factor):
#    downsampled[i] = mean(headangle[downsampling_factor*i:downsampling_factor*(i+1)])
#headangle = downsampled
print("Binning spikes...")
nbins = len(trackingtimes) // downsampling_factor
binnedspikes = zeros((len(cellnames), nbins))
sgood = zeros(len(binnedspikes[:,0]))<1
for i in range(len(cellnames)):
  spikes = ravel((cellspikes[0])[i])
  for j in range(len(spikes)):
    # note 1ms binning means that number of ms from start is the correct index
    tt = int(floor(  (spikes[j] - startt)/float(binsize)  ))
    if(tt>nbins-1 or tt<0): # check if outside bounds of the awake time
      continue
    binnedspikes[i,tt] += 1 # add a spike to the thing

  ## check if neuron has enough spikes
  if(sum(binnedspikes[i,:])<thresholdforneuronstokeep):
      sgood[i] = False
      continue
  ## remove neurons that are not tuned to head direction
  if i not in [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,38,47]:
      sgood[i] = False
      continue

# Remove neurons that have less total spikes than a threshold 
# Also remove those that are not tuned to head direction
binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]

# Remove nan items
whiches = np.isnan(headangle)
headangle = headangle[~whiches]
binnedspikes = binnedspikes[:,~whiches]

# Choose an interval of the total observed time 
path = headangle[offset:offset+T]
binnedspikes = binnedspikes[:,offset:offset+T]
print("How many times are there more than one spike:", sum((binnedspikes>1)*1))
if LIKELIHOOD_MODEL == "bernoulli":
    # Set spike counts to 0 or 1
    binnedspikes = (binnedspikes>0)*1
if (sum(isnan(path)) > 0):
    print("\nXXXXXXXXX\nXXXXXXXXX\nXXXXXXXXX\nThere are NAN values in path\nXXXXXXXXX\nXXXXXXXXX\nXXXXXXXXX")

N = len(cellnames) #51 with cutoff at 1000 spikes
print("N:",N)
y_spikes = binnedspikes
print("mean(y_spikes)",mean(y_spikes))
print("mean(y_spikes>0)",mean(y_spikes[y_spikes>0]))
# Spike distribution evaluation
spike_count = np.ndarray.flatten(binnedspikes)
print("How many times are there more than one spike:", sum(spike_count>1))
print("Percentage of those bins with 1 that actually have more than 1 spike:", sum(spike_count>1) / sum(spike_count>0)) #len(spike_count[spike_count>0]))
# Remove zero entries:
#spike_count = spike_count[spike_count>0]
plt.figure()
plt.hist(spike_count, bins=np.arange(0,int(max(spike_count))+1)-0.5, log=True, color=plt.cm.viridis(0.3))
plt.ylabel("Number of bins")
plt.xlabel("Spike count")
plt.title("Spike histogram")
plt.xticks(range(0,int(max(spike_count)),1))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-spike-histogram-log.png")

# Find average observed spike count
number_of_X_bins = 50
bins = np.linspace(min(path)-0.000001, max(path) + 0.0000001, num=number_of_X_bins + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])
observed_spikes = zeros((N, number_of_X_bins))
# Observed spike average and estimated tuning curves
for i in range(N):
    for x in range(number_of_X_bins):
        timesinbin = (path>bins[x])*(path<bins[x+1])
        if(sum(timesinbin)>0):
            observed_spikes[i,x] = mean( y_spikes[i, timesinbin] )

## Plot average observed spike count
fig, ax = plt.subplots()
foo_mat = ax.matshow(y_spikes) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("y spikes")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-y-spikes.png")

######################
# Covariance kernels #
######################

def exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def squared_exponential_covariance(x1,x2, sigma, delta):
    if COVARIANCE_KERNEL_KX == "periodic":
        distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    elif COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

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
    #start = time.time()
    K_xg = np.zeros((T,N_inducing_points))
    for x1 in range(T):
        for x2 in range(N_inducing_points):
            K_xg[x1,x2] = squared_exponential_covariance(X_estimate[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T
    #stop = time.time()
    #print("Making Kxg            :", stop-start)

    #start = time.time()
    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    #stop = time.time()
    #print("Making small/tempmatrx:", stop-start)

    # yf_term ##########
    ####################
    #start = time.time()
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.log(1 + np.exp(F_estimate)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, F_estimate) - np.exp(F_estimate))
    #stop = time.time()
    #print("yf term               :", stop-start)

    # f prior term #####
    ####################
    #start = time.time()
    f_prior_term_1 = sigma_n**-2 * np.trace(np.matmul(F_estimate, F_estimate.T))
    fK = np.matmul(F_estimate, tempmatrix)
    fKf = np.matmul(fK, F_estimate.T)
    f_prior_term_2 = - sigma_n**-2 * np.trace(fKf)

    f_prior_term = - 0.5 * (f_prior_term_1 + f_prior_term_2)
    #stop = time.time()
    #print("f prior term          :", stop-start)

    # logdet term ######
    ####################
    # My variant: 
    #logdet_term = - 0.5 * N * np.log(np.linalg.det(Kx_inducing))
    # Wu variant:
    #start = time.time()
    logDetS1 = -np.log(np.linalg.det(smallinverse))-np.log(np.linalg.det(K_gg))+np.log(sigma_n)*(T-N_inducing_points)
    logdet_term = - 0.5 * N * logDetS1
    #stop = time.time()
    #print("logdet term            :", stop-start)

    # x prior term #####
    ####################
    #start = time.time()
    xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)
    #stop = time.time()
    #print("X prior term          :", stop-start)

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

def scaling(offset):
    scaled_estimate = X_estimate + offset
    return just_fprior_term(scaled_estimate)

########################
# Covariance functions #
########################

# Inducing points based on where the X actually are
x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 
K_gg_plain = np.zeros((N_inducing_points,N_inducing_points))
for x1 in range(N_inducing_points):
    for x2 in range(N_inducing_points):
        K_gg_plain[x1,x2] = squared_exponential_covariance(x_grid_induce[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
#fig, ax = plt.subplots()
#foo_mat = ax.matshow(K_gg_plain, cmap=plt.cm.Blues)
#fig.colorbar(foo_mat, ax=ax)
#plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-kgg.png")

K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
K_gg_inverse = np.linalg.inv(K_gg)

K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)

######################
# Initialize X and F #
######################
# xinitialize
offset = 0
r = 0.3
X_initial = 2 * np.ones(T)
X_initial = path
#X_initial = np.load("X_estimate.npy")
#np.pi* 
# offset + r * path + (1-r)*np.pi + 0.2*np.sin(np.linspace(0,10*np.pi,T))
#np.pi * np.ones(T)
#np.sqrt(path)
#np.pi*np.ones(T)
#r * path + (1-r)*np.pi
#2*np.pi*np.random.random(T)
X_estimate = np.copy(X_initial)

# finitialize
F_initial = np.sqrt(y_spikes)
F_estimate = np.copy(F_initial)

## Plot initial f
fig, ax = plt.subplots()
foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("Initial f")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-initial-f.png")

collected_estimates = np.zeros((N_iterations, T))

x_posterior_no_la(X_estimate)

plt.figure()
plt.title("X estimates across iterations")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.ylim((0,2*np.pi))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-collected-estimates.png")
# Find best offset
#print("\n\nFind best offset X for sigma =",sigma_n)
#initial_offset = 0
#scaling_optimization_result = optimize.minimize(scaling, initial_offset, method = "L-BFGS-B", options = {'disp':True})
#best_offset = scaling_optimization_result.x
#X_estimate = X_estimate + best_offset
#plt.plot(X_estimate, label = "Scaled")
#plt.legend()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-collected-estimates.png")
plt.show()
### EM algorithm: Find f given X, then X given f.
for iteration in range(N_iterations):
    print("\nIteration", iteration)
    if sigma_n > 0.85: #1e-8:
        sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
    print("Sigma2:", sigma_n)
    print("L value at path for this sigma:",x_posterior_no_la(path))
    K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
    K_gg_inverse = np.linalg.inv(K_gg)

    K_xg_prev = np.zeros((T,N_inducing_points))
    for x1 in range(T):
        for x2 in range(N_inducing_points):
            K_xg_prev[x1,x2] = squared_exponential_covariance(X_estimate[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
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
    fig, ax = plt.subplots()
    foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("F estimate")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-F-estimate.png")
    plt.clf()
    plt.close()

    # Find next X estimate, that can be outside (0,2pi)
    print("Finding next X estimate...")

    # Attempt to regularize by adding noise to estimate to shake it up
    #if iteration < 5:
    #    X_estimate += sigma_n/5 * np.random.random(T)

    optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", options = {'disp':True}) #jac=x_jacobian_decoupled_la, 
    X_estimate = optimization_result.x

    # Find best offset
    print("\n\nFind best offset X for sigma =",sigma_n)
    initial_offset = 0
    scaling_optimization_result = optimize.minimize(scaling, initial_offset, method = "L-BFGS-B", options = {'disp':True})
    best_offset = scaling_optimization_result.x
    X_estimate = X_estimate + best_offset
    print("Best offset:", best_offset)

    #plt.plot(X_estimate, label='Best offset')

    plt.figure()
    plt.title("X estimates across iterations")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    collected_estimates[iteration] = np.transpose(X_estimate)
    for i in range(int(iteration+1)):
        plt.plot(collected_estimates[i], label="Estimate") #"%s" % i
    #plt.legend()
    plt.ylim((0,2*np.pi))
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-collected-estimates.png")
    plt.clf()
    plt.close()
    np.save("X_estimate", X_estimate)

# Final estimate
plt.figure()
plt.title("Final estimate")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.plot(X_estimate, label='Estimate')
plt.legend()
plt.ylim((0,2*np.pi))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-final.png")
plt.show()

###########################
# Flipped 
X_flipped = - X_estimate + 2*mean(X_estimate)

plt.figure()
plt.title("Flipped estimate")
plt.plot(X_initial, label='Initial')
plt.plot(path, color="black", label='True X')
#plt.plot(X_estimate, label='Estimate')
plt.plot(X_flipped, label='Flipped')
plt.legend()
plt.ylim((0,2*np.pi))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-EM-flipped.png")
