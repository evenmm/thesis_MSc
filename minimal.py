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
T = 50
N = 100

P = 1 # Dimensions of latent variable 
N_plotgridpoints = N # Number of grid points for plotting f posterior only 
sigma_f_fit = 2 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.15 # Scale for the tuning curve GP that is fitted. 0.3
sigma_n = 0.001 # Assumed variance of observations for the GP that is fitted. 10e-5
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
print("Likelihood model:",LIKELIHOOD_MODEL)
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
sigma_x = 2*np.pi # Variance of X for K_t # 2*np.pi
delta_x = 10 # Scale of X for K_t
sigma_n_x = 0.1 # This one only belongs to Kt
N_iterations = 10

######################################
## Generate data for simple example ##
######################################
# Generate X
path = np.pi + np.pi * np.sin(np.linspace(0,2*np.pi,T))

# Generate F
tuningwidth = 1 # width of tuning (in radians)
biasterm = -2 # Average H outside tuningwidth -4
tuningcovariatestrength = 4.*tuningwidth # H value at centre of tuningwidth 6*tuningwidth
neuronpeak = [(i+0.5)*2.*pi/N for i in range(N)]

def exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def squared_exponential_covariance(x1,x2, sigma, delta):
    if COVARIANCE_KERNEL_KX == "periodic":
        distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    elif COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=N_plotgridpoints + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])

print("Setting y_spikes equal to true spike probabilities") # print("Generating spikes")
true_f = np.zeros((N, T))
for i in range(N):
    for t in range(T):
        if COVARIANCE_KERNEL_KX == "periodic":
            distancefrompeaktopathpoint = min([ abs(neuronpeak[i]+2.*pi-path[t]),  abs(neuronpeak[i]-path[t]),  abs(neuronpeak[i]-2.*pi-path[t]) ])
        elif COVARIANCE_KERNEL_KX == "nonperiodic":
            distancefrompeaktopathpoint = abs(neuronpeak[i]-path[t])
        Ht = biasterm
        if(distancefrompeaktopathpoint < tuningwidth):
            Ht = biasterm + tuningcovariatestrength * (1-distancefrompeaktopathpoint/tuningwidth)
        true_f[i,t] = Ht

#########################
## Likelihood functions #
#########################

# p(X|F), faster version? 01.04
def simplified_x_posterior(X_estimate):
    Kx = make_Kx_with_sigma(T,X_estimate)
    Kx_inverse = np.linalg.inv(Kx)

    # f prior term #####
    ####################
    # This only enforces smoothness in f
    # We need to evaluate f at its current place and then compare to true f, or y if we don't have f
    f_prior_term = 0
    # "fy term" ########
    ####################
    current_f = np.zeros((N,T))
    for i in range(N):
        for t in range(T):
            if COVARIANCE_KERNEL_KX == "periodic":
                distancefrompeaktopathpoint = min([ abs(neuronpeak[i]+2.*pi-X_estimate[t]),  abs(neuronpeak[i]-X_estimate[t]),  abs(neuronpeak[i]-2.*pi-X_estimate[t]) ])
            elif COVARIANCE_KERNEL_KX == "nonperiodic":
                distancefrompeaktopathpoint = abs(neuronpeak[i]-X_estimate[t])
            Ht = biasterm
            if(distancefrompeaktopathpoint < tuningwidth):
                Ht = biasterm + tuningcovariatestrength * (1-distancefrompeaktopathpoint/tuningwidth)
            current_f[i,t] = Ht

        f_prior_term += - 0.5 * np.dot(np.dot(current_f[i], Kx_inverse), current_f[i].T)
    #fy_term = LAMBDA * np.sum((current_f - F_estimate)**2)

    # logdet term ######
    ####################
    logdet_term = - N/2 * np.log(np.linalg.det(Kx))

    # x prior term ######
    ####################
    x_prior_term = -0.5 * np.dot(np.dot(X_estimate.T, K_t_inverse), X_estimate)

    posterior_loglikelihood = logdet_term + f_prior_term + x_prior_term #+ fy_term
    return - posterior_loglikelihood

def differentiated_SE_covariance(X_estimate, t1):
    # t is the element of X that we differentiate with regards to.
    x_t1 = X_estimate[t1]
    Kx_differentiated_wrt_xt = np.zeros((T,T))
    for t2 in range(T):
        x_t2 = X_estimate[t2]
        Kx_differentiated_wrt_xt[t1, t2] = - (x_t1 - x_t2)/delta_f_fit**2 * sigma_f_fit * np.exp(- (x_t1 - x_t2)**2 /(2*delta_f_fit**2))
        Kx_differentiated_wrt_xt[t2, t1] = - Kx_differentiated_wrt_xt[t1, t2]
    Kx_differentiated_wrt_xt[t1, t1] = 0
    return Kx_differentiated_wrt_xt

def x_jacobian(X_estimate):
    Kx = make_Kx_with_sigma(T,X_estimate)
    Kx_inverse = np.linalg.inv(Kx)

    # f prior term #####
    ####################
    f_prior_term = np.zeros(T)
    logdet_term = np.zeros(T)
    for t in range(T):
        Kx_differentiated_wrt_xt = differentiated_SE_covariance(X_estimate, t)
        #print(Kx_differentiated_wrt_xt) # This one looks bad
        f_i = F_estimate[i]
        midmatrix = - np.matmul(np.matmul(Kx_inverse, Kx_differentiated_wrt_xt), Kx_inverse)
        f_prior_term[t] = - 0.5 * np.dot(np.dot(f_i, midmatrix), f_i.T)

        # logdet term ######
        ####################
        logdet_term[t] = - N/2 * np.trace(np.matmul(Kx_inverse, Kx_differentiated_wrt_xt))

    # x prior term ######
    ####################
    x_prior_term = - np.dot(K_t_inverse, X_estimate)

    jacobian = logdet_term + f_prior_term + x_prior_term
    return - jacobian

########################
# Covariance functions #
########################
def make_Kx_with_sigma(T, X_estimate):
    Kx_fit_at_observations = np.zeros((T,T))
    for x1 in range(T):
        for x2 in range(T):
            Kx_fit_at_observations[x1,x2] = squared_exponential_covariance(X_estimate[x1],X_estimate[x2], sigma_f_fit, delta_f_fit)
    # By adding sigma_epsilon on the diagonal, we assume noise and make the covariance matrix positive semidefinite
    Kx_fit_at_observations = Kx_fit_at_observations  + np.identity(T)*sigma_n
    return Kx_fit_at_observations

K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_x, delta_x)
K_t += sigma_n_x * np.identity(T)
K_t_inverse = np.linalg.inv(K_t)

## Set F value to truth
F_estimate = true_f

## Plot true f
plt.figure()
plt.xlabel("Head direction")
color_idx = np.linspace(0, 1, N)
plt.ylabel("True f")
for i in range(N):
    plt.plot(F_estimate[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-true-tuning.pdf",format="pdf")

print("Test L value for different values of X")
print("path\n",simplified_x_posterior(path))
print("path + 0.1sinus\n",simplified_x_posterior(path + 0.1*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.2sinus\n",simplified_x_posterior(path + 0.2*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.3sinus\n",simplified_x_posterior(path + 0.3*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.4sinus\n",simplified_x_posterior(path + 0.4*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.5sinus\n",simplified_x_posterior(path + 0.5*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.6sinus\n",simplified_x_posterior(path + 0.6*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.7sinus\n",simplified_x_posterior(path + 0.7*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 1.0sinus\n",simplified_x_posterior(path + 1.0*np.sin(np.linspace(0,2*np.pi,T))))
print("Random start\n",simplified_x_posterior(2*np.pi*np.random.random(T)))

# Set initial X
X_initial = 2*np.pi*np.random.random(T)
#np.linspace(np.pi/4,3/2*np.pi,T) 
#np.repeat(np.pi,T) 
#path + 0.6*np.sin(np.linspace(0,2*np.pi,T)) 
#2*np.pi*np.random.random(T)
print("initial\n",simplified_x_posterior(X_initial))
X_estimate = X_initial # path + 0.6*np.sin(np.linspace(0,2*np.pi,T)) #np.random.random((N,T)) #path + 0.3*np.sin(np.linspace(0,2*np.pi,T))

#print("Testing gradient")
#grad = x_jacobian(X_estimate)
#print(grad)

# Plot initial X compared to truth
plt.figure()
plt.title("Initial X")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.legend()
plt.show()

# Iterate. Find next X estimate, that can be outside (0,2pi)
for iteration in range(N_iterations):
    print("Finding next X estimate...")
    optimization_result = optimize.minimize(simplified_x_posterior, X_estimate, method = "L-BFGS-B", options = {'disp':True}) # jac=x_jacobian, 
    X_estimate = optimization_result.x

    plt.figure()
    plt.title("Inferred X estimate")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    plt.plot(X_estimate, label='Estimated')
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-X-estimate.png")
    plt.legend()
    plt.show()

plt.figure()
plt.title("Inferred X estimate")
plt.plot(path, color="black", label='True X')
plt.plot(X_initial, label='Initial')
plt.plot(X_estimate, label='Estimated')
plt.legend()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-inference.png")
plt.show()
