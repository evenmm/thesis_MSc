from scipy import *
import scipy.io
import scipy.ndimage
import numpy as np
import scipy.optimize as sp
import numpy.random
import matplotlib
#matplotlib.use('Agg') # When running on cluster, plots cannot be shown and this must be used
import matplotlib.pyplot as plt
import time
import sys
plt.rc('image', cmap='viridis')
from scipy import optimize
numpy.random.seed(13)

###############
# Conventions #
###############
# All likelihood, gradient functions etc return negative versions
# Minimize the negative likelihood.

##############
# Parameters #
##############
offset = 1000 # Starting point in observed X values
T = 1000
P = 1 # Dimensions of latent variable 
sigma_fit = 8 # Variance for the GP that is fitted. 8
delta_fit = 0.3 # Scale for the GP that is fitted. 0.3
sigma_epsilon_fit = 0.2 # Assumed variance of observations for the GP that is fitted. 10e-5
gridpoints = 50 # Number of grid points
TOLERANCE_X = 0.001 # for X posterior
LIKELIHOOD_MODEL = 1 # 1. Bernoulli 2. Poisson
sigma_t = 1 # Variance for K_t
delta_t = 0.1 # Scale for K_t

##############################
# Data fetch and definitions #
##############################

name = sys.argv[1] #'Mouse28-140313_stuff_BS0030_awakedata.mat'

thresholdforneuronstokeep = 1000 # number of spikes to be considered useful

mat = scipy.io.loadmat(name)
headangle = ravel(array(mat['headangle']))
cellspikes = array(mat['cellspikes'])
cellnames = array(mat['cellnames'])
trackingtimes = ravel(array(mat['trackingtimes']))

## make matrix of spikes y_spikes
startt = min(trackingtimes)
binsize = mean(trackingtimes[1:]-trackingtimes[:(-1)])
nbins = len(trackingtimes)
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

  ## check if actvitity is ok
  if(sum(binnedspikes[i,:])<thresholdforneuronstokeep):
      sgood[i] = False
      continue

binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]

# Remove nan items
whiches = np.isnan(headangle)
headangle = headangle[~whiches]
binnedspikes = binnedspikes[:,~whiches]

# Select part of data to be able to make X
true_path = headangle[offset:offset+T]
binnedspikes = binnedspikes[:,offset:offset+T]
binnedspikes = (binnedspikes>0)*1 #Reset to ones 

N = len(cellnames) #51 with cutoff at 1000 spikes
print("T:",T)
print("N:",N)
print("How many times are there more than one spike:", sum((binnedspikes>1)*1))
y_spikes = binnedspikes
print("mean(y_spikes)",mean(y_spikes))

def exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def gaussian_periodic_covariance(x1,x2, sigma, delta):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return sigma * exp(-distancesquared/(2*delta))

def gaussian_NONPERIODIC_covariance(x1,x2, sigma, delta):
    distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

###############################
## Inference of tuning curves #
###############################

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize.
def f_loglikelihood_bernoulli(f_i):
    likelihoodterm = sum( np.multiply(y_i, (f_i - np.log(1+np.exp(f_i)))) + np.multiply((1-y_i), np.log(1- np.divide(np.exp(f_i), 1 + np.exp(f_i)))))
    priorterm = - 0.5*np.dot(f_i, np.dot(Kx_fit_at_observations_inverse, f_i))
    return - (likelihoodterm + priorterm)
def f_jacobian_bernoulli(f_i):
    e_tilde = np.divide(exp(f_i), 1 + exp(f_i))
    f_derivative = y_i - e_tilde - np.dot(Kx_fit_at_observations_inverse, f_i)
    return - f_derivative
def f_hessian_bernoulli(f_i):
    e_plain_fraction = np.divide(exp(f_i), (1 + exp(f_i))**2)
    f_hessian = - np.diag(e_plain_fraction) - Kx_fit_at_observations_inverse 
    return - f_hessian

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize.
def f_loglikelihood_poisson(f_i):
    return 0
def f_jacobian_poisson(f_i):
    return 0
def f_hessian_poisson(f_i):
    return 0

## Optimization of f given X
#def find_f_hat(N, T, X_estimate, y_spikes, likelihood_model, Kx_fit_at_observations_inverse):
#    f_tuning_curve = np.zeros(shape(y_spikes)) #np.sqrt(y_spikes) # Initialize f values
#    if likelihood_model == 1: # Bernoulli
#        for i in range(N):
#            y_i = y_spikes[i]
#           optimization_result = optimize.minimize(f_loglikelihood_bernoulli, f_tuning_curve[i], jac=f_jacobian_bernoulli, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
#            f_tuning_curve[i] = optimization_result.x
#    elif likelihood_model == 2: # Poisson
#        for i in range(N):
#            y_i = y_spikes[i]
#            optimization_result = optimize.minimize(f_loglikelihood_poisson, f_tuning_curve[i], jac=f_jacobian_poisson, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
#            f_tuning_curve[i] = optimization_result.x
#    return f_tuning_curve

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize.
def x_jacobian():

    return 0

def x_loglikelihood(X):
    if LIKELIHOOD_MODEL == 1: # Bernoulli, equation 4.26
        fy_term = sum(np.multiply(f_hat, y_spikes) - np.log(1 + np.exp(f_hat)))
    elif LIKELIHOOD_MODEL == 2: # Poisson, equation 4.43
        fy_term = sum(np.multiply(f_hat, y_spikes) - np.exp(f_hat))
    xTKt = np.dot(np.transpose(X), K_t_inverse)
    prior_term_X = - 0.5 * np.dot(xTKt, X)
    prior_term_f = 0
    for ii in range(N):
        fTKx = np.dot(np.transpose(f_hat[ii]), Kx_fit_at_observations_inverse)
        prior_term_f += np.dot(fTKx, f_hat[ii])
    posterior_likelihood = fy_term + prior_term_f + prior_term_X #+ determinant_term + prior_term_f 
    return - posterior_likelihood
#        e_bernoulli = np.divide(exp(f), (1 + exp(f))**2)
#        determinant_term = 0
#        for i in range(N):
#            W_i = np.diag(e_bernoulli[i])
#            prod = np.dot(W_i, Kx_fit_at_observations) + np.identity(T)
#            determinant_term += - 0.5 * np.log(np.linalg.det(prod))
#        determinant_term = - 0.5 * sum( )
#    elif likelihood_model == 2: # Poisson
#        fy_term = sum(np.multiply(f_hat, y_spikes)) - sum(np.exp(f_hat))
#        determinant_term = 0
#    for i in range(N):
#        fTKx = np.dot(np.transpose(f[i]), Kx_fit_at_observations_inverse)
#        prior_term_f += - 0.5 * np.dot(fTKx, f[i])

###########################
# EM Inference of X and f #
###########################
def make_Kx(T, X_estimate):
    Kx_fit_at_observations = np.zeros((T,T))
    for x1 in range(T):
        for x2 in range(T):
            Kx_fit_at_observations[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],X_estimate[x2], sigma_fit, delta_fit)
    # By adding sigma_epsilon on the diagonal, we assume noise and make the covariance matrix positive semidefinite
    Kx_fit_at_observations = Kx_fit_at_observations  + np.identity(T)*sigma_epsilon_fit
    return Kx_fit_at_observations

K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_t, delta_t)
K_t_inverse = np.linalg.inv(K_t)

X_estimate = np.pi * np.ones(T)
X_loglikelihood_old = 0
X_loglikelihood_new = np.inf
iteration = -1
while abs(X_loglikelihood_new - X_loglikelihood_old) > TOLERANCE_X:
    X_loglikelihood_old = X_loglikelihood_new
    iteration += 1
    print("\nEM Iteration:", iteration, "\nX estimate:", X_estimate[0:5],"\n")
    Kx_fit_at_observations = make_Kx(T, X_estimate)
    Kx_fit_at_observations_inverse = np.linalg.inv(Kx_fit_at_observations)

    # Find f hat given X
    print("Finding f hat...")
    f_tuning_curve = np.zeros(shape(y_spikes)) #np.sqrt(y_spikes) # Initialize f values
    if LIKELIHOOD_MODEL == 1: # Bernoulli
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_bernoulli, f_tuning_curve[i], jac=f_jacobian_bernoulli, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
            f_tuning_curve[i] = optimization_result.x
    elif LIKELIHOOD_MODEL == 2: # Poisson
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_poisson, f_tuning_curve[i], jac=f_jacobian_poisson, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
            f_tuning_curve[i] = optimization_result.x
    #f_hat = find_f_hat(N, T, X_estimate, y_spikes, LIKELIHOOD_MODEL, Kx_fit_at_observations_inverse)
    f_hat = f_tuning_curve
    print("Finding next X estimate...")
    optimization_result = optimize.minimize(x_loglikelihood, X_estimate, method = "Newton-CG", options = {'disp':True})
    X_estimate = optimization_result.x
    X_loglikelihood_new = optimization_result.fun 


#################################################
# Find posterior prediction of log tuning curve #
#################################################
bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=gridpoints + 1)
x_grid = 0.5*(bins[:(-1)]+bins[1:])
f_values_observed = f_hat

print("Making spatial covariance matrice: Kx crossover")
Kx_crossover = np.zeros((T,gridpoints))
for x1 in range(T):
    for x2 in range(gridpoints):
        Kx_crossover[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],x_grid[x2], sigma_fit, delta_fit)
#fig, ax = plt.subplots()
#kx_cross_mat = ax.matshow(Kx_crossover, cmap=plt.cm.Blues)
#fig.colorbar(kx_cross_mat, ax=ax)
#plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-kx_crossover.png")
Kx_crossover_T = np.transpose(Kx_crossover)
print("Making spatial covariance matrice: Kx grid")
Kx_grid = np.zeros((gridpoints,gridpoints))
for x1 in range(gridpoints):
    for x2 in range(gridpoints):
        Kx_grid[x1,x2] = gaussian_periodic_covariance(x_grid[x1],x_grid[x2], sigma_fit, delta_fit)
fig, ax = plt.subplots()
kxmat = ax.matshow(Kx_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-kx_grid.png")

# Infer mean on the grid
pre = np.zeros((N,T))
mu_posterior = np.zeros((N, gridpoints))
for i in range(N):
    pre[i] = np.dot(Kx_fit_at_observations_inverse, f_values_observed[i])
    mu_posterior[i] = np.dot(Kx_crossover_T, pre[i])
# Calculate standard deviations
sigma_posterior = (Kx_grid) - np.dot(Kx_crossover_T, np.dot(Kx_fit_at_observations_inverse, Kx_crossover))
fig, ax = plt.subplots()
sigma_posteriormat = ax.matshow(sigma_posterior, cmap=plt.cm.Blues)
fig.colorbar(sigma_posteriormat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-sigma_posterior.png")

###############################################
# Plot tuning curve with confidence intervals #
###############################################
standard_deviation = np.sqrt(np.diag(sigma_posterior))
upper_confidence_limit = mu_posterior + 1.96*standard_deviation
lower_confidence_limit = mu_posterior - 1.96*standard_deviation

h_estimate = np.exp(mu_posterior)/ (1 + np.exp(mu_posterior))
h_upper_confidence_limit = np.exp(upper_confidence_limit) / (1 + np.exp(upper_confidence_limit))
h_lower_confidence_limit = np.exp(lower_confidence_limit) / (1 + np.exp(lower_confidence_limit))

## Find observed firing rate
observed_spikes = np.zeros((N, gridpoints))
for i in range(N):
    for x in range(gridpoints):
        timesinbin = (X_estimate>bins[x])*(X_estimate<bins[x+1])
        if(sum(timesinbin)>0): 
            observed_spikes[i,x] = np.mean( y_spikes[i, timesinbin] )
#        else:
#            print("No observations of X between",bins[x],"and",bins[x+1],".")

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
for n4 in range(1): #range(N//4):
    plt.figure(figsize=(10,8))
    neuron = np.array([[0,1],[2,3]])
    neuron = neuron + 4*n4
    for i in range(2):
        for j in range(2):
            plt.subplot(2,2,i*2+j+1)
            plt.plot(x_grid, observed_spikes[neuron[i,j],:], color="#cfb302")
            plt.plot(x_grid, h_estimate[neuron[i,j],:], color=colors[0]) 
            plt.plot(x_grid, h_upper_confidence_limit[neuron[i,j],:], "--", color=colors[0])
            plt.plot(x_grid, h_lower_confidence_limit[neuron[i,j],:], "--", color=colors[0])
            plt.ylim(0.,1.)
            plt.title(neuron[i,j]+1)
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"hd-fitted-tuning"+str(n4+1)+".png")

## plot actual head direction together with estimate
plt.figure(figsize=(10,2))
plt.plot(true_path, '.', color='black', markersize=2.)
plt.plot(X_estimate, '.', color=plt.cm.viridis(0.5), markersize=2.)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference.png")
plt.show()

