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
#from hd_dataload import *


# All likelihood, gradient functions etc return negative versions
# Minimize the negative likelihood.

##############
# Parameters #
##############
offset = 1000 # Starting point in observed X values
T = 100
P = 1 # Dimensions of latent variable 
sigma_f_fit = 8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.3 # Scale for the tuning curve GP that is fitted. 0.3
sigma_epsilon_f_fit = 0.2 # Assumed variance of observations for the GP that is fitted. 10e-5
gridpoints = 50 # Number of grid points
TOLERANCE_X = 0.1 # for X posterior
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
print("Likelihood model:",LIKELIHOOD_MODEL)
INFERENCE_METHOD = 3 # 1. No LA. 2. Standard LA. 3. Decoupled LA
sigma_x = 2 # Variance of X for K_t
delta_x = 10 # Scale of X for K_t
N_iterations = 10
plottruth = True

def exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def gaussian_periodic_covariance(x1,x2, sigma, delta):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return sigma * exp(-distancesquared/(2*delta))

def gaussian_NONPERIODIC_covariance(x1,x2, sigma, delta):
    distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

######################################
## Generate data for simple example ##
######################################
N = 1

# Generative path for X:
#sigma_path = 1 # Variance
#delta_path = 50 # Scale 
#Kt = np.zeros((T, T)) 
#for t1 in range(T):
#    for t2 in range(T):
#        Kt[t1,t2] = exponential_covariance(t1,t2, sigma_path, delta_path)
#path = numpy.random.multivariate_normal(np.zeros(T), Kt)
#path = np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi
path = 2*np.sin([2*np.pi*t/T for t in range(T)])
# plot path
if plottruth:
    plt.figure()#(figsize=(10,2))
    plt.plot(path, '.', color='black', markersize=1.)
    plt.xlabel("Time")
    plt.ylabel("x value")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-path.pdf",format="pdf")

# Define spike rates
def f(x): 
    return 2-6*x**2
# Plot f and h
if plottruth:
    plt.figure()
    xplotgrid = np.linspace(-2,2,100)
    #plt.plot(xplotgrid,f(xplotgrid))
    plt.plot(xplotgrid,np.exp(f(xplotgrid))/(1+np.exp(f(xplotgrid))), color="blue")
    plt.title("Spike rate h in dark blue")
    plt.ylim(0,1)

# Generate y_spikes, Bernoulli
true_f = f(path)
rates = np.exp(true_f)/(1+np.exp(true_f)) # h tuning curve values
y_spikes = np.array([np.random.binomial(1, rates)])

###############################
## Inference of tuning curves #
###############################

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize. Equation (4.17)++
def f_loglikelihood_bernoulli(f_i): # Psi
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.log(1+np.exp(f_i))) # Corrected 16.03 from sum( np.multiply(y_i, (f_i - np.log(1+np.exp(f_i)))) + np.multiply((1-y_i), np.log(1- np.divide(np.exp(f_i), 1 + np.exp(f_i)))))
    priorterm = - 0.5*np.dot(np.transpose(f_i), np.dot(Kx_fit_at_observations_inverse, f_i))
    return - (likelihoodterm + priorterm)
def f_jacobian_bernoulli(f_i):
    e_plain = np.divide(np.exp(f_i), 1 + np.exp(f_i))
    f_derivative = y_i - e_plain - np.dot(Kx_fit_at_observations_inverse, f_i)
    return - f_derivative
def f_hessian_bernoulli(f_i):
    e_tilde = np.divide(np.exp(f_i), (1 + np.exp(f_i))**2)
    f_hessian = - np.diag(e_tilde) - Kx_fit_at_observations_inverse 
    return - f_hessian

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize.
def f_loglikelihood_poisson(f_i):
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.exp(f_i)) 
    priorterm = - 0.5*np.dot(np.transpose(f_i), np.dot(Kx_fit_at_observations_inverse, f_i))
    return - (likelihoodterm + priorterm)
def f_jacobian_poisson(f_i):
    e_poiss = np.exp(f_i)
    f_derivative = y_i - e_poiss - np.dot(Kx_fit_at_observations_inverse, f_i)
    return - f_derivative
def f_hessian_poisson(f_i):
    e_poiss = np.exp(f_i)
    f_hessian = - np.diag(e_poiss) - Kx_fit_at_observations_inverse
    return - f_hessian

# NEGATIVE Loglikelihood and gradient. minimize to maximize.
def x_loglikelihood_decoupled_la(X):
    # yf_term
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, f_hat) - np.log(1 + np.exp(f_hat)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, f_hat) - np.exp(f_hat))
    # f prior term
    f_prior_term = 0
    for ii in range(N):
        fTKx = np.dot(np.transpose(f_hat[ii]), Kx_fit_at_observations_inverse)
        f_prior_term += np.dot(fTKx, f_hat[ii])
    # det term
    det_term = 0
    for ii in range(N):
        S_inverse = np.diag(np.exp(f_hat[ii]))
        tempmatrix = np.matmul(S_inverse, Kx_fit_at_observations) + np.identity(T) 
        det_term += - 0.5 * np.log(np.linalg.det(tempmatrix))
    # x prior term
    xTKt = np.dot(np.transpose(X), K_t_inverse)
    x_prior_term = - 0.5 * np.dot(xTKt, X)

    posterior_loglikelihood = yf_term + det_term + f_prior_term + x_prior_term
    return - posterior_loglikelihood

def x_jacobian_decoupled_la(X):
    
    return 0

###########################
# EM Inference of X and f #
###########################
def make_Kx(T, X_estimate):
    Kx_fit_at_observations = np.zeros((T,T))
    for x1 in range(T):
        for x2 in range(T):
            Kx_fit_at_observations[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],X_estimate[x2], sigma_f_fit, delta_f_fit)
    # By adding sigma_epsilon on the diagonal, we assume noise and make the covariance matrix positive semidefinite
    Kx_fit_at_observations = Kx_fit_at_observations  + np.identity(T)*sigma_epsilon_f_fit
    return Kx_fit_at_observations

K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)

# Initialize X
#X_estimate = np.pi * np.ones(T)
X_estimate = path

X_loglikelihood_old = 0
X_loglikelihood_new = np.inf
### INFERENCE OF X
for iteration in range(N_iterations):
#while abs(X_loglikelihood_new - X_loglikelihood_old) > TOLERANCE_X:
    plt.figure()
    plt.plot(path, color="blue")
    plt.plot(X_estimate)
    plt.show()
    print("Logikelihood improvement:", - (X_loglikelihood_new - X_loglikelihood_old))
    X_loglikelihood_old = X_loglikelihood_new
    print("\nEM Iteration:", iteration, "\nX estimate:", X_estimate[0:5],"\n")
    Kx_fit_at_observations = make_Kx(T, X_estimate)
    Kx_fit_at_observations_inverse = np.linalg.inv(Kx_fit_at_observations)

    # Find f hat given X
    print("Finding f hat...")
    f_tuning_curve = np.zeros(shape(y_spikes)) #np.sqrt(y_spikes) # Initialize f values
    if LIKELIHOOD_MODEL == "bernoulli":
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_bernoulli, f_tuning_curve[i], jac=f_jacobian_bernoulli, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
            f_tuning_curve[i] = optimization_result.x
    elif LIKELIHOOD_MODEL == "poisson":
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_poisson, f_tuning_curve[i], jac=f_jacobian_poisson, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
            f_tuning_curve[i] = optimization_result.x #f_hat = find_f_hat(N, T, X_estimate, y_spikes, LIKELIHOOD_MODEL, Kx_fit_at_observations_inverse)
    f_hat = f_tuning_curve
    plt.figure()
    plt.plot(f_hat[0])
    plt.plot(true_f, color="blue")
    plt.show()
    # Find next X estimate, that can be outside (0,2pi)
    print("Finding next X estimate...")
    if INFERENCE_METHOD == 3:
        optimization_result = optimize.minimize(x_loglikelihood_decoupled_la, X_estimate, jac=x_jacobian_decoupled_la, method = "L-BFGS-B", options = {'disp':True})
    X_estimate = optimization_result.x
    # Reshape X to be in (0,2pi)
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
        Kx_crossover[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],x_grid[x2], sigma_f_fit, delta_f_fit)
#fig, ax = plt.subplots()
#kx_cross_mat = ax.matshow(Kx_crossover, cmap=plt.cm.Blues)
#fig.colorbar(kx_cross_mat, ax=ax)
#plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-kx_crossover.png")
Kx_crossover_T = np.transpose(Kx_crossover)
print("Making spatial covariance matrice: Kx grid")
Kx_grid = np.zeros((gridpoints,gridpoints))
for x1 in range(gridpoints):
    for x2 in range(gridpoints):
        Kx_grid[x1,x2] = gaussian_periodic_covariance(x_grid[x1],x_grid[x2], sigma_f_fit, delta_f_fit)
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

plt.figure()
plt.plot(x_grid, observed_spikes[0,:], color="#cfb302")
plt.plot(x_grid, h_estimate[0,:], color=colors[0]) 
plt.plot(x_grid, h_upper_confidence_limit[0,:], "--", color=colors[0])
plt.plot(x_grid, h_lower_confidence_limit[0,:], "--", color=colors[0])
plt.ylim(0.,1.)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"hd-fitted-tuning.png")

#for n4 in range(1): #range(N//4):
#    plt.figure(figsize=(10,8))
#    neuron = np.array([[0,1],[2,3]])
#    neuron = neuron + 4*n4
#    for i in range(2):
#        for j in range(2):
#            plt.subplot(2,2,i*2+j+1)
#            plt.plot(x_grid, observed_spikes[neuron[i,j],:], color="#cfb302")
#            plt.plot(x_grid, h_estimate[neuron[i,j],:], color=colors[0]) 
#            plt.plot(x_grid, h_upper_confidence_limit[neuron[i,j],:], "--", color=colors[0])
#            plt.plot(x_grid, h_lower_confidence_limit[neuron[i,j],:], "--", color=colors[0])
#            plt.ylim(0.,1.)
#            plt.title(neuron[i,j]+1)
#    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"hd-fitted-tuning"+str(n4+1)+".png")

## plot actual head direction together with estimate
plt.figure(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=2.)
plt.plot(X_estimate, '.', color=plt.cm.viridis(0.5), markersize=2.)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference.png")
plt.show()

