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

##############
# Parameters #
##############x
T = 80 # 2000
N = 100
N_iterations = 10
sigma_n = 1.2 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.95 # Learning rate by which we multiply sigma_n at every iteration

N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 100 # Number of grid points for plotting f posterior only 
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
TUNINGCURVE_DEFINITION = "bumps" # "triangles" "bumps"
sigma_f_fit = 2 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.7 # Scale for the tuning curve GP that is fitted. 0.3
sigma_x = 6 # Variance of X for K_t
delta_x = 10 # Scale of X for K_t
P = 1 # Dimensions of latent variable 
USE_OFFSET_AFTER_ITERATION_NUMBER = 2
USE_SCALING_AFTER_ITERATION_NUMBER = 3
NOISE_REGULARIZATION = False
FLIP_AFTER_TWO_ITERATIONS = False
GIVEN_TRUE_F = False
OPTIMIZE_HYPERPARAMETERS = False
GRADIENT_FLAG = False # Choose to use gradient or not

print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("Using gradient?", GRADIENT_FLAG)
print("True tuning curve shape:", TUNINGCURVE_DEFINITION)
print("Noise regulation:",NOISE_REGULARIZATION)
print("Initial sigma_n:", sigma_n)
print("Learning rate:", lr)
print("T:", T, "\n")
print("N:", N, "\n")
if FLIP_AFTER_TWO_ITERATIONS:
    print("NBBBB!!! We're flipping the estimate after the second iteration in line 600.")

######################################
## Data generation                  ##
######################################

def exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def squared_exponential_covariance(x1,x2, sigma, delta):
    if COVARIANCE_KERNEL_KX == "periodic":
        distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    elif COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

if TUNINGCURVE_DEFINITION == "triangles":
    tuningwidth = 1 # width of tuning (in radians)
    biasterm = -2 # Average H outside tuningwidth -4
    tuningcovariatestrength = np.linspace(0.5*tuningwidth,10.*tuningwidth, N) # H value at centre of tuningwidth 6*tuningwidth
    neuronpeak = [(i+0.5)*2.*pi/N for i in range(N)]

bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=N_plotgridpoints + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])

# Generative path for X:
sigma_path = 1 # Variance
delta_path = 50 # Scale 
Kt = np.zeros((T, T)) 
for t1 in range(T):
    for t2 in range(T):
        Kt[t1,t2] = exponential_covariance(t1,t2, sigma_path, delta_path)

#path = np.pi + numpy.random.multivariate_normal(np.zeros(T), Kt)
#path[40:70] = path[40] + np.sin(np.linspace(0,4*np.pi,30))
#path[70:100] = np.linspace(path[70],np.pi,30)
path = np.pi + numpy.random.multivariate_normal(np.zeros(T), Kt)
#path = np.linspace(0,2*np.pi,T)
# np.pi + np.pi*np.sin(np.linspace(0,10*np.pi,T))
# np.linspace(0,2*np.pi,T)
#np.array(np.pi + 1*np.pi*np.sin([2*np.pi*t/T for t in range(T)]))
#numpy.random.multivariate_normal(np.zeros(T), Kt)
#np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi

## Generate spike data from a Bernoulli GLM (logistic regression) 
# True tuning curves are defined here
bumplocations = 2*np.pi*np.random.random(N)
bumpwidths = 0.01 + 0.5*np.random.random(N)
def bumptuningfunction(x, i): 
    return squared_exponential_covariance(x, bumplocations[i], 2, bumpwidths[i])

if TUNINGCURVE_DEFINITION == "triangles":
    print("Generating spikes.")
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
    print("Generating spikes")
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

## Plot true f
plt.figure()
plt.xlabel("Head direction")
color_idx = np.linspace(0, 1, N)
plt.ylabel("True f")
for i in range(N):
    plt.plot(true_f[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-true-f-curves.png")
plt.show()

## Plot true f
fig, ax = plt.subplots()
foo_mat = ax.matshow(true_f) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("True f")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-true-f-mnatrix.png")
plt.clf()
plt.close()

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
    xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)

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
#X_initial = np.load("X_estimate.npy")
#np.pi* 
# offset + r * path + (1-r)*np.pi + 0.2*np.sin(np.linspace(0,10*np.pi,T))
#np.pi * np.ones(T)
#np.sqrt(path)
#np.pi*np.ones(T)
#r * path + (1-r)*np.pi
#2*np.pi*np.random.random(T)
X_estimate = X_initial

# finitialize
F_initial = np.sqrt(y_spikes) # np.sqrt(y_spikes) true_f
F_estimate = F_initial

## Plot initial f
fig, ax = plt.subplots()
foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("Initial f")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-initial-f.png")
plt.clf()
plt.close()

collected_estimates = np.zeros((N_iterations, T))

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
    # Initialize f guess at every iteration
    #F_estimate = true_f #np.sqrt(y_spikes) 

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
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-F-estimate.png")
    fig, ax = plt.subplots()
    foo_mat = ax.matshow(F_estimate-true_f) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("Difference between F estimate and truth")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-F-difference.png")
    #plt.show()
    plt.clf()
    plt.close()

    # Find next X estimate, that can be outside (0,2pi)
    print("Finding next X estimate...")

    # Attempt to regularize by adding noise to estimate to shake it up
    if iteration < 5:
        X_estimate += sigma_n/5 * np.random.random(T)

    optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", options = {'disp':True}) #jac=x_jacobian_decoupled_la, 
    X_estimate = optimization_result.x

    # Find best offset
    print("\n\nFind best offset X for sigma =",sigma_n)
    initial_offset = 0
    scaling_optimization_result = optimize.minimize(scaling, initial_offset, method = "L-BFGS-B", options = {'disp':True})
    best_offset = scaling_optimization_result.x
    if iteration<(N_iterations-1):
        X_estimate = X_estimate + best_offset #0.5*
    else:
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
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-collected-estimates.png")
    #plt.show()
    np.save("X_estimate", X_estimate)
    plt.clf()
    plt.clf()
    plt.close()

# Final estimate
plt.figure()
plt.title("Final estimate")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.plot(X_estimate, label='Estimate')
plt.legend()
plt.ylim((0,2*np.pi))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-final.png")
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
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-flipped.png")
