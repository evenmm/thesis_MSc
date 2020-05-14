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
##############
T = 100
N = 100
N_iterations = 100
sigma_n = 0.7 # Assumed variance of observations for the GP that is fitted. 10e-5

P = 1 # Dimensions of latent variable 
N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 100 # Number of grid points for plotting f posterior only 
sigma_f_fit = 2 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.7 # Scale for the tuning curve GP that is fitted. 0.3
lr = 0.95 # Learning rate by which we multiply sigma_n at every iteration
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
print("Likelihood model:",LIKELIHOOD_MODEL)
COVARIANCE_KERNEL_KX = "periodic" # "periodic" "nonperiodic"<f
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
GRADIENT_FLAG = False # Choose to use gradient or not
print("\nUsing gradient?", GRADIENT_FLAG, "\n\n")
sigma_x = 6 # Variance of X for K_t
delta_x = 10 # Scale of X for K_t

tuningwidth = 1 # width of tuning (in radians)
biasterm = -2 # Average H outside tuningwidth -4
tuningcovariatestrength = 4.*tuningwidth # H value at centre of tuningwidth 6*tuningwidth
neuronpeak = [(i+0.5)*2.*pi/N for i in range(N)]
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
bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=N_plotgridpoints + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])

# Generative path for X:
sigma_path = 1 # Variance
delta_path = 50 # Scale 
Kt = np.zeros((T, T)) 
for t1 in range(T):
    for t2 in range(T):
        Kt[t1,t2] = exponential_covariance(t1,t2, sigma_path, delta_path)

path = np.pi + numpy.random.multivariate_normal(np.zeros(T), Kt)
path[40:70] = path[40] + np.sin(np.linspace(0,4*np.pi,30))
path[70:100] = np.linspace(path[70],np.pi,30)
# np.pi + np.pi*np.sin(np.linspace(0,10*np.pi,T))
# np.linspace(0,2*np.pi,T)
#np.array(np.pi + 1*np.pi*np.sin([2*np.pi*t/T for t in range(T)]))
#numpy.random.multivariate_normal(np.zeros(T), Kt)
#np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi

## Generate spike data from a Bernoulli GLM (logistic regression) 
# True tuning curves are defined here
def samplefromBernoulli(H):
    p = exp(H)/(1.+exp(H)) ## p for the Logit link function
    return 1.0*(rand()<p)

print("Setting y_spikes equal to true spike probabilities") # print("Generating spikes")
true_f = np.zeros((N, T))
y_spikes = np.zeros((N, T))
for i in range(N):
    for t in range(T):
        distancefrompeaktopathpoint = min([ abs(neuronpeak[i]+2.*pi-path[t]),  abs(neuronpeak[i]-path[t]),  abs(neuronpeak[i]-2.*pi-path[t]) ])
        Ht = biasterm
        if(distancefrompeaktopathpoint < tuningwidth):
            Ht = biasterm + tuningcovariatestrength * (1-distancefrompeaktopathpoint/tuningwidth)
        #y_spikes[i,t] = samplefromBernoulli(Ht) ## Spike with probability e^H/1+e^H
        true_f[i,t] = Ht
        if LIKELIHOOD_MODEL == "bernoulli":
            y_spikes[i,t] = exp(Ht)/(1.+exp(Ht))
        elif LIKELIHOOD_MODEL == "poisson":
            y_spikes[i,t] = exp(Ht)

"""
## Plot true f
plt.figure()
plt.xlabel("Head direction")
color_idx = np.linspace(0, 1, N)
plt.ylabel("True f")
for i in range(N):
    plt.plot(true_f[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.show()
"""
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
            K_xg[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
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






########################
# Covariance functions #
########################

# Inducing points based on where the X actually are
x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 
K_gg_plain = np.zeros((N_inducing_points,N_inducing_points))
for x1 in range(N_inducing_points):
    for x2 in range(N_inducing_points):
        K_gg_plain[x1,x2] = gaussian_periodic_covariance(x_grid_induce[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
#fig, ax = plt.subplots()
#kgg_cross_mat = ax.matshow(K_gg_plain, cmap=plt.cm.Blues)
#fig.colorbar(kgg_cross_mat, ax=ax)
#plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-kgg.png")

K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)

# Initialize X
offset = 0
r = 0.3
X_initial = np.pi*np.ones(T)
# offset + r * path + (1-r)*np.pi + 0.2*np.sin(np.linspace(0,10*np.pi,T))
#np.pi * np.ones(T)
#np.sqrt(path)
#np.pi*np.ones(T)
#r * path + (1-r)*np.pi
#2*np.pi*np.random.random(T)
X_estimate = X_initial
# F_estimate = np.sqrt(y_spikes)
F_estimate = true_f

K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
K_gg_inverse = np.linalg.inv(K_gg)

tempsigma = sigma_n
sigma_n = 1e-4
print("Test L value for different X, with sigma == 1e-4")
print("path\n",x_posterior_no_la(path))
print("path + 0.1*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 0.1*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.2*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 0.2*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.3*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 0.3*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.4*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 0.4*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.5*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 0.5*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.6*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 0.6*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.7*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 0.7*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 1.0*np.sin(np.linspace(0,2*np.pi,T))\n",x_posterior_no_la(path + 1.0*np.sin(np.linspace(0,2*np.pi,T))))
print("Random start\n",x_posterior_no_la(2*np.pi*np.random.random(T)), "\n")
sigma_n = tempsigma

plt.figure()
plt.title("Inferred X estimate")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-inducing-no-la-X-estimate.png")
### INFERENCE OF X USING DECOUPLED LAPLACE APPROXIMATION. Input: Obervations y and initial guess X0
for iteration in range(N_iterations):
    print("Iteration", iteration)
    if sigma_n > 1e-8:
        sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
    print("Sigma2:", sigma_n)
    print("L value at path for this sigma:",x_posterior_no_la(path))
    K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
    K_gg_inverse = np.linalg.inv(K_gg)

    previous_X = X_estimate
    K_xg_prev = np.zeros((T,N_inducing_points))
    for x1 in range(T):
        for x2 in range(N_inducing_points):
            K_xg_prev[x1,x2] = gaussian_periodic_covariance(X_estimate[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
    K_gx_prev = K_xg_prev.T

    #print("Finding f hat...")
    print("Setting f hat equal to true f")
    # Initialize f values
    f_tuning_curve = true_f #np.sqrt(y_spikes) 
    """
    if LIKELIHOOD_MODEL == "bernoulli":
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_bernoulli, f_tuning_curve[i], jac=f_jacobian_bernoulli, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
            f_tuning_curve[i] = optimization_result.x
    elif LIKELIHOOD_MODEL == "poisson":
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_poisson, f_tuning_curve[i], jac=f_jacobian_poisson, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
            f_tuning_curve[i] = optimization_result.x 
    F_estimate = f_tuning_curve
    """

    # Find next X estimate, that can be outside (0,2pi)
    print("Finding next X estimate...")
    optimization_result = optimize.minimize(x_posterior_no_la, X_estimate, method = "L-BFGS-B", options = {'disp':True}) #jac=x_jacobian_decoupled_la, 
    X_estimate = optimization_result.x

    plt.plot(X_estimate, label='Estimated')
    plt.legend()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-inducing-no-la-X-estimate.png")

plt.figure()
plt.title("Inferred X estimate")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.plot(X_estimate, label='Estimated')
plt.legend()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference.png")
plt.show()


