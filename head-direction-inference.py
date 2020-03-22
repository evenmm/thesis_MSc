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
#from hd_dataload import *

##############
# Parameters #
##############
P = 1 # Dimensions of latent variable 
N_inducing_points = 25 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 50 # Number of grid points for plotting f posterior only 
sigma_f_fit = 8 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.7 # Scale for the tuning curve GP that is fitted. 0.3
sigma_n = 0.2 # Assumed variance of observations for the GP that is fitted. 10e-5
TOLERANCE_X = 0.1 # for X posterior
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
print("Likelihood model:",LIKELIHOOD_MODEL)
INFERENCE_METHOD = 3 # 1. No LA. 2. Standard LA. 3. Decoupled LA
sigma_x = 2*np.pi # Variance of X for K_t
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
# Generative path for X:
#sigma_path = 1 # Variance
#delta_path = 50 # Scale 
#Kt = np.zeros((T, T)) 
#for t1 in range(T):
#    for t2 in range(T):
#        Kt[t1,t2] = exponential_covariance(t1,t2, sigma_path, delta_path)
#path = numpy.random.multivariate_normal(np.zeros(T), Kt)
#path = np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi
T = 100
N = 10
tuningwidth = 1 # width of tuning (in radians)
biasterm = -2 # Average H outside tuningwidth -4
tuningcovariatestrength = 4.*tuningwidth # H value at centre of tuningwidth 6*tuningwidth
neuronpeak = [(i+0.5)*2.*pi/N for i in range(N)]
number_of_bins = 50

bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=number_of_bins + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])

path = np.linspace(0,2*np.pi,T)
#path = np.array(np.pi + 1*np.pi*np.sin([2*np.pi*t/T for t in range(T)]))

## Generate spike data from a Bernoulli GLM (logistic regression) 
# True tuning curves are defined here
def samplefromBernoulli(H):
    p = exp(H)/(1.+exp(H)) ## p for the Logit link function
    return 1.0*(rand()<p)

print("Setting y_spikes equal to true spike probabilities") # print("Generating spikes")
y_spikes = np.zeros((N, T))
for i in range(N):
    for t in range(T):
        distancefrompeaktopathpoint = min([ abs(neuronpeak[i]+2.*pi-path[t]),  abs(neuronpeak[i]-path[t]),  abs(neuronpeak[i]-2.*pi-path[t]) ])
        Ht = biasterm
        if(distancefrompeaktopathpoint < tuningwidth):
            Ht = biasterm + tuningcovariatestrength * (1-distancefrompeaktopathpoint/tuningwidth)
        #y_spikes[i,t] = samplefromBernoulli(Ht) ## Spike with probability e^H/1+e^H
        if LIKELIHOOD_MODEL == "bernoulli":
            y_spikes[i,t] = exp(Ht)/(1.+exp(Ht))
        elif LIKELIHOOD_MODEL == "poisson":
            y_spikes[i,t] = exp(Ht)/(1.+exp(Ht))
true_spike_probability = np.zeros((N, number_of_bins))
true_spike_rate = np.zeros((N,number_of_bins))
for i in range(N):
    for x in range(number_of_bins):
        distancefrompeaktopathpoint = min([ abs(neuronpeak[i]+2.*pi-evaluationpoints[x]),  abs(neuronpeak[i]-evaluationpoints[x]),  abs(neuronpeak[i]-2.*pi-evaluationpoints[x]) ])
        Ht = biasterm
        if(distancefrompeaktopathpoint < tuningwidth):
            Ht = biasterm + tuningcovariatestrength * (1-distancefrompeaktopathpoint/tuningwidth)
        true_spike_probability[i,x] = np.exp(Ht)/(1.+np.exp(Ht))
        true_spike_rate[i,x] = np.exp(Ht)

## Plot true tuning curves alone
plt.figure()
plt.xlabel("Head direction")
color_idx = np.linspace(0, 1, N)
if LIKELIHOOD_MODEL == "bernoulli":
    plt.ylabel("Spike probability")
    for i in range(N):
        plt.plot(evaluationpoints, true_spike_probability[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
elif LIKELIHOOD_MODEL == "poisson":
    plt.ylabel("Spike rate")
    for i in range(N):
        plt.plot(evaluationpoints, true_spike_rate[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-true-tuning.pdf",format="pdf")

#########################
## Likelihood functions #
#########################

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize. Equation (4.17)++
def f_loglikelihood_bernoulli(f_i): # Psi
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.log(1+np.exp(f_i))) # Corrected 16.03 from sum( np.multiply(y_i, (f_i - np.log(1+np.exp(f_i)))) + np.multiply((1-y_i), np.log(1- np.divide(np.exp(f_i), 1 + np.exp(f_i)))))
    priorterm = - 0.5*np.dot(f_i.T, np.dot(Kx_fit_at_observations_inverse, f_i))
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
    priorterm_1 = -0.5/sigma_n**2 * np.dot(f_i.T, f_i)
    fT_k = np.dot(f_i, K_xg_prev)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev, K_xg_prev.T))
    priorterm_2 = -0.5/sigma_n**2 * np.dot(np.dot(fT_k, smallinverse), fT_k.T)
    return - (likelihoodterm + priorterm_1 + priorterm_2)
    # All together version_
    #if LIKELIHOOD_MODEL == "bernoulli":
    #    likelihoodterm = sum(np.multiply(y_spikes, f_tuning_curve) - np.log(1 + np.exp(f_tuning_curve)))
    #elif LIKELIHOOD_MODEL == "poisson":
    #    likelihoodterm = sum(np.multiply(y_spikes, f_tuning_curve) - np.exp(f_tuning_curve))
    #priorterm_1 = -0.5/sigma_n**2 * np.trace(f_tuning_curve.T, f_tuning_curve)
    #fT_k = np.matmul(f_tuning_curve, K_xg_prev)
    #smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev, K_xg_prev.T))
    #priorterm_2 = -0.5/sigma_n**2 * np.trace(np.matmul(np.matmul(fT_k, smallinverse), fT_k.T))
    #return - (likelihoodterm + priorterm_1 + priorterm_2)

def f_jacobian_poisson(f_i):
    e_poiss = np.exp(f_i)
    f_derivative = y_i - e_poiss - np.dot(Kx_fit_at_observations_inverse, f_i)
    return - f_derivative
def f_hessian_poisson(f_i):
    e_poiss = np.exp(f_i)
    f_hessian = - np.diag(e_poiss) - Kx_fit_at_observations_inverse
    return - f_hessian

# Matrix inversion trick
def smartinverse(Kxg):
    # Uses Matrix inversion lemma where Z is an identity matrix times sigma^2
    # The output of this function is the inverse of (Kx + sigma^2I) = (Kxg Kgg^-1 Kgx + sigma^2I)

    return sigma**-2 * np.identity(N) - sigma**-2 
    # Should these boys be sparse?

# L function: negative Loglikelihood
def x_posterior_loglikelihood_decoupled_la(U): # Analog to logmargli_gplvm_se_sor_la_decouple.m
    f_prior_term = 0
    logdet_term = 0
    X_estimate = np.dot(K_t_squareroot, U) # works for 1D
    
    K_xg = np.zeros((T,N_inducing_points))
    for x1 in range(T):
        for x2 in range(N_inducing_points):
            K_xg[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T

    K_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2 * np.identity(T)

    # To find A
    Q = np.matmul(K_xg_prev, Kgg_inv_half)
    lowdim_inverse = np.linalg.inv( 1/sigma_n**2 * np.identity(N_inducing_points) + np.matmul(Q.T, Q) )
    inverse_K_prev = 1/sigma_n**2 * np.identity(T) - 1/sigma_n**2 * np.matmul( np.matmul(Q, lowdim_inverse) , Q.T )

    # To find A at X
    Q = np.matmul(K_xg, Kgg_inv_half)
    lowdim_inverse = np.linalg.inv( 1/sigma_n**2 * np.identity(N_inducing_points) + np.matmul(Q.T, Q) )
    inverse_K_current = 1/sigma_n**2 * np.identity(T) - 1/sigma_n**2 * np.matmul( np.matmul(Q, lowdim_inverse) , Q.T )

    f_at_X = np.zeros((N,T))
    for i in range(N):
        f_i = f_hat[i]
        W_i = np.diag(np.exp(f_i))

        # Finding A = W_i + K_x^-1
        A = W_i + inverse_K_prev
        # Finding A(X)
        A_at_X = W_i + inverse_K_current

        A_times_f = np.dot(A, f_i)
        f_at_X[i] = np.linalg.solve(A_at_X, A_times_f)

        # f prior term
        fTKx = np.dot(f_at_X[i].T), inverse_K_current)
        f_prior_term += np.dot(fTKx, f_at_X[i])

        # logdet term
        tempmatrix = np.matmul(W_i, K_inducing) + np.identity(T) 
        logdet_term += - 0.5 * np.log(np.linalg.det(tempmatrix))

    # yf_term
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, f_at_X) - np.log(1 + np.exp(f_at_X)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, f_at_X) - np.exp(f_at_X))

    # x prior term
    xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)

    posterior_loglikelihood = yf_term + logdet_term + f_prior_term + x_prior_term
    return - posterior_loglikelihood

def x_jacobian_decoupled_la(X):
    # f1term
    f1term = 0
    # f2term
    f2term = 0
    # logdetterm
    logdetterm = 0
    # f_prior_term
    f_prior_term = 0
    # x_prior_term
    x_prior_term = 0
    jacobian = f1term + f2term + logdetterm + f_prior_term + x_prior_term
    return - jacobian

########################
# Covariance functions #
########################
def make_Kx_with_sigma(T, X_estimate):
    Kx_fit_at_observations = np.zeros((T,T))
    for x1 in range(T):
        for x2 in range(T):
            Kx_fit_at_observations[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],X_estimate[x2], sigma_f_fit, delta_f_fit)
    # By adding sigma_epsilon on the diagonal, we assume noise and make the covariance matrix positive semidefinite
    Kx_fit_at_observations = Kx_fit_at_observations  + np.identity(T)*sigma_n
    return Kx_fit_at_observations

# Inducing points based on where the X actually are
x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 
K_gg = np.zeros((N_inducing_points,N_inducing_points))
for x1 in range(N_inducing_points):
    for x2 in range(N_inducing_points):
        K_gg[x1,x2] = gaussian_periodic_covariance(x_grid_induce[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
K_gg += sigma_n*np.identity(N_inducing_points) # This is unexpected but Wu does the same thing
fig, ax = plt.subplots()
kgg_cross_mat = ax.matshow(K_gg, cmap=plt.cm.Blues)
fig.colorbar(kgg_cross_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-kgg.png")
K_gg_inverse = np.linalg.inv(K_gg) # Since we are adding sigma in the algorithm, we should not do it here

Kgg_half = scipy.linalg.sqrtm(K_gg)
Kgg_inv_half = scipy.linalg.sqrtm(K_gg_inverse)

# Change this from X to U
K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)
K_t_squareroot = scipy.linalg.sqrtm(K_t)
K_t_inverse_squareroot = scipy.linalg.sqrtm(K_t_inverse)

# Initialize X
#X_estimate = np.pi * np.ones(T)
X_estimate = path + 0.3*np.sin(np.linspace(0,2*np.pi,T))
F_estimate = np.sqrt(y_spikes)

X_loglikelihood_old = 0
X_loglikelihood_new = np.inf
plt.show()
### INFERENCE OF X USING DECOUPLED LAPLACE APPROXIMATION. Input: Obervations y and initial guess X0
for iteration in range(N_iterations):
    previous_X = X_estimate
    K_xg_prev = np.zeros((T,N_inducing_points))
    for x1 in range(T):
        for x2 in range(N_inducing_points):
            K_xg_prev[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)

    plt.figure()
    plt.title("X estimate")
    plt.plot(path, color="blue")
    plt.plot(X_estimate)
    plt.show()
    print("Logikelihood improvement:", - (X_loglikelihood_new - X_loglikelihood_old))
    X_loglikelihood_old = X_loglikelihood_new
    print("\nEM Iteration:", iteration, "\nX estimate:", X_estimate[0:5],"\n")
    Kx_fit_at_observations = make_Kx_with_sigma(T, X_estimate)
    Kx_fit_at_observations_inverse = np.linalg.inv(Kx_fit_at_observations)

    print("Finding f hat...")
    f_tuning_curve = np.sqrt(y_spikes) # Initialize f values

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
    plt.title("f estimate")
    for i in range(N):
        plt.plot(f_hat[i])
    plt.show()
    # Find next X estimate, that can be outside (0,2pi)
    print("Finding next X estimate...")
    optimization_result = optimize.minimize(x_posterior_loglikelihood_decoupled_la, X_estimate, method = "L-BFGS-B", options = {'disp':True}) #jac=x_jacobian_decoupled_la, 
    X_estimate = optimization_result.x
    F_estimate = f_hat
    # Reshape X to be in (0,2pi)?
    X_loglikelihood_new = optimization_result.fun 











#################################################
# Find posterior prediction of log tuning curve #
#################################################
bins = np.linspace(min(X) -0.000001, max(X) + 0.0000001, num=N_plotgridpoints + 1) # (-0.000001, 2.*np.pi+0.0000001)
x_grid = 0.5*(bins[:(-1)]+bins[1:]) # for plotting with uncertainty
f_values_observed = f_hat

print("Making spatial covariance matrice: Kx crossover")
Kx_crossover = np.zeros((T,N_plotgridpoints))
for x1 in range(T):
    for x2 in range(N_plotgridpoints):
        Kx_crossover[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],x_grid[x2], sigma_f_fit, delta_f_fit)
#fig, ax = plt.subplots()
#kx_cross_mat = ax.matshow(Kx_crossover, cmap=plt.cm.Blues)
#fig.colorbar(kx_cross_mat, ax=ax)
#plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-kx_crossover.png")
Kx_crossover_T = Kx_crossover.T
print("Making spatial covariance matrice: Kx grid")
Kx_grid = np.zeros((N_plotgridpoints,N_plotgridpoints))
for x1 in range(N_plotgridpoints):
    for x2 in range(N_plotgridpoints):
        Kx_grid[x1,x2] = gaussian_periodic_covariance(x_grid[x1],x_grid[x2], sigma_f_fit, delta_f_fit)
fig, ax = plt.subplots()
kxmat = ax.matshow(Kx_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference-kx_grid.png")

# Infer mean on the grid
pre = np.zeros((N,T))
mu_posterior = np.zeros((N, N_plotgridpoints))
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
observed_spikes = np.zeros((N, N_plotgridpoints))
for i in range(N):
    for x in range(N_plotgridpoints):
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

