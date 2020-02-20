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
from gplvm import *
numpy.random.seed(13)

##############
# Parameters #
##############
offset = 1000 # Starting point in observed X values
T = 1000
sigma_fit = 8 # Variance for the GP that is fitted. 8
delta_fit = 0.3 # Scale for the GP that is fitted. 0.3
sigma_epsilon_fit = 0.2 # Assumed variance of observations for the GP that is fitted. 10e-5
X_dim = 50 # Number of grid points
TOLERANCE_f = 1 # for Newton's method
TOLERANCE_X = 0.1 # for X posterior
LIKELIHOOD = "bernoulli"

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

# Function to check if loglikelihood is increasing for f_tuning_curve
def loglikelihoodfunction(y_i, f_i):
    if (np.shape(y_i) != np.shape(f_i)):
        print("Size mismatch between y_i and f_i in loglikelihood function!")
    likelihoodterm = sum( np.multiply(y_i, (f_i - np.log(1+np.exp(f_i)))) + np.multiply((1-y_i), np.log(1- np.divide(np.exp(f_i), 1 + np.exp(f_i)))))
    priorterm = - 0.5*np.dot(f_i, np.dot(Kx_fit_at_observations_inverse, f_i))
    return likelihoodterm + priorterm 


def find_f_hat(N, T, path, y_spikes, likelihood):
    ####################################################
    ## Inference of tuning curves and latent variable ##
    ####################################################
    ## Using Gaussian Processes ########################
    ####################################################

    Kx_fit_at_observations = np.zeros((T,T))
    for x1 in range(T):
        for x2 in range(T):
            Kx_fit_at_observations[x1,x2] = gaussian_periodic_covariance(x_values_observed[x1],x_values_observed[x2], sigma_fit, delta_fit)
    # By adding sigma_epsilon on the diagonal, we assume noise and make the covariance matrix positive semidefinite
    Kx_fit_at_observations = Kx_fit_at_observations  + np.identity(T)*sigma_epsilon_fit
    Kx_fit_at_observations_inverse = np.linalg.inv(Kx_fit_at_observations)

    ## Finding f hat using Newton's method
    f_tuning_curve = np.zeros(shape(y_spikes)) # Initialize f values
    f_convergence_plot_for_neuron_0 = zeros((100,T)) ## Here we store f values of neuron 0 at every time at each iteration step
    iteration = 0
    while (True):
        iteration += 1
        if iteration<4: 
            learning_rate = 0.1
        elif iteration<10:
            learning_rate = 0.1
        elif iteration<15:
            learning_rate = 0.1
        else: 
            learning_rate = 0.1 
        old_likelihoods = np.array([loglikelihoodfunction(y_spikes[i],f_tuning_curve[i]) for i in range(N)])
        for i in range(N): # See equtaion 4.36 in overleaf
            # if LIKELIHOOD=="bernoulli"
            e_tilde = np.divide(exp(f_tuning_curve[i]), 1 + exp(f_tuning_curve[i]))
            e_plain_fraction = np.divide(exp(f_tuning_curve[i]), (1 + exp(f_tuning_curve[i]))**2)
            f_derivative = y_spikes[i] - e_tilde - np.dot(Kx_fit_at_observations_inverse, f_tuning_curve[i])
            f_hessian = - np.diag(e_plain_fraction) - Kx_fit_at_observations_inverse 
            # This formulation of Newton's method is slow: 
            #new_f_i = f_tuning_curve[i] - learning_rate * np.dot(f_derivative,np.linalg.inv(f_hessian))
            # Instead we do like this (fast): 
            delta_f_i = np.linalg.solve(f_hessian, - learning_rate * f_derivative)
            new_f_i = f_tuning_curve[i] + delta_f_i
            f_tuning_curve[i] = new_f_i
            if (i==0):
                f_convergence_plot_for_neuron_0[iteration-1] = f_tuning_curve[i]
        new_likelihoods = np.array([loglikelihoodfunction(y_spikes[i],f_tuning_curve[i]) for i in range(N)])
        print("\nNewton Iteration",iteration)
        print("Biggest likelihood difference", max(new_likelihoods - old_likelihoods)) #The biggest likelihood improvement across all neurons
        if (max(abs(new_likelihoods - old_likelihoods)) < TOLERANCE_f):
            break
        old_likelihoods = new_likelihoods
    return f_tuning_curve

##################
# Inference of X #
##################
X_estimate = np.zeros(T)
X_loglikelihood_old = -2*TOLERANCE_X
X_loglikelihood_new = 0
while abs(X_loglikelihood_new - X_loglikelihood_old) > TOLERANCE_X:
    X_loglikelihood_old = X_loglikelihood_new
    f_hat = find_f_hat(N, T, X_estimate, y_spikes, likelihood=LIKELIHOOD)

    
    



    X_loglikelihood_new = ...



#################################################
# Find posterior prediction of log tuning curve #
#################################################
bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=X_dim + 1)
x_grid = 0.5*(bins[:(-1)]+bins[1:])
f_values_observed = f_hat

print("Making spatial covariance matrice: Kx crossover")
Kx_crossover = np.zeros((T,X_dim))
for x1 in range(T):
    for x2 in range(X_dim):
        Kx_crossover[x1,x2] = gaussian_periodic_covariance(x_values_observed[x1],x_grid[x2], sigma_fit, delta_fit)
fig, ax = plt.subplots()
kx_cross_mat = ax.matshow(Kx_crossover, cmap=plt.cm.Blues)
fig.colorbar(kx_cross_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-kx_crossover.png")
Kx_crossover_T = np.transpose(Kx_crossover)
print("Making spatial covariance matrice: Kx grid")
Kx_grid = np.zeros((X_dim,X_dim))
for x1 in range(X_dim):
    for x2 in range(X_dim):
        Kx_grid[x1,x2] = gaussian_periodic_covariance(x_grid[x1],x_grid[x2], sigma_fit, delta_fit)
fig, ax = plt.subplots()
kxmat = ax.matshow(Kx_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-kx_grid.png")

# Infer mean on the grid
pre = np.zeros((N,T))
mu_posterior = np.zeros((N, X_dim))
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
observed_spikes = zeros((N, X_dim))
for i in range(N):
    for x in range(X_dim):
        timesinbin = (X_estimate>bins[x])*(X_estimate<bins[x+1])
        if(sum(timesinbin)>0):
            observed_spikes[i,x] = mean( y_spikes[i, timesinbin] )
        else:
            print("No observations of X between",bins[x],"and",bins[x+1],".")

for n4 in range(N//4):
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
plt.show()

## plot actual head direction together with estimate
plt.figure(figsize=(10,2))
plt.plot(true_path, '.', color='black', markersize=1.)
plt.plot(true_path, '.', color=plt.cm.viridis(0), markersize=1.)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference.png")

