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

P = 1 # Dimensions of latent variable 
N_inducing_points = 30 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 100 # Number of grid points for plotting f posterior only 
sigma_f_fit = 2 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.15 # Scale for the tuning curve GP that is fitted. 0.3
sigma_n = 0.001 # Assumed variance of observations for the GP that is fitted. 10e-5
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
print("Likelihood model:",LIKELIHOOD_MODEL)
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
GRADIENT = False # Choose to use gradient or not
print("\nUsing gradient?", GRADIENT, "\n\n")
sigma_x = 2*np.pi # Variance of X for K_t
delta_x = 10 # Scale of X for K_t
N_iterations = 10

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

##################
## Generate data #
##################
# Generate X
path = np.linspace(0,2*np.pi,T)
#path = np.array(np.pi + 1*np.pi*np.sin([2*np.pi*t/T for t in range(T)]))

bins = np.linspace(-0.000001, 2.*np.pi+0.0000001, num=N_plotgridpoints + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])

## Generate spike data from a Bernoulli GLM (logistic regression) 
# True tuning curves are defined here
def samplefromBernoulli(H):
    p = exp(H)/(1.+exp(H)) ## p for the Logit link function
    return 1.0*(rand()<p)

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
            Ht = biasterm + tuningcovariatestrength * (1-distancefrompeaktopathpoint/tuningwidth)
        true_f[i,t] = Ht
        if LIKELIHOOD_MODEL == "bernoulli":
            y_spikes[i,t] = exp(Ht)/(1.+exp(Ht))
        elif LIKELIHOOD_MODEL == "poisson":
            y_spikes[i,t] = exp(Ht)

## Plot true f
plt.figure()
plt.xlabel("Head direction")
color_idx = np.linspace(0, 1, N)
plt.ylabel("True f")
for i in range(N):
    plt.plot(true_f[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-hd-true-tuning.pdf",format="pdf")

#########################
## Likelihood functions #
#########################   

# Posterior of X
def simplified_x_posterior(X_estimate):
    Kx = make_Kx_with_sigma(T,X_estimate)
    Kx_inverse = np.linalg.inv(Kx)

    # f prior term #####
    ####################
    f_prior_term = 0
    for i in range(N):
        f_prior_term += - 0.5 * np.dot(np.dot(true_f[i], Kx_inverse), true_f[i].T)

    # yf_term ##########
    ####################
    if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
        yf_term = sum(np.multiply(y_spikes, true_f) - np.log(1 + np.exp(true_f)))
    elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
        yf_term = sum(np.multiply(y_spikes, true_f) - np.exp(true_f))
    
    # logdet term ######
    ####################
    logdet_term = - N/2 * np.log(np.linalg.det(Kx))

    # x prior term #####
    ####################
    x_prior_term = -0.5 * np.dot(np.dot(X_estimate.T, K_t_inverse), X_estimate)

    posterior_loglikelihood = logdet_term + f_prior_term + x_prior_term + yf_term
    return - posterior_loglikelihood

# Differentiating K_x
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

# Jacobian
def x_jacobian(X_estimate):
    Kx = make_Kx_with_sigma(T,X_estimate)
    Kx_inverse = np.linalg.inv(Kx)

    # f prior term #####
    ####################
    f_prior_term = np.zeros(T)

    # logdet term ######
    ####################
    logdet_term = np.zeros(T)

    for t in range(T):
        Kx_differentiated_wrt_xt = differentiated_SE_covariance(X_estimate, t)
        #print(Kx_differentiated_wrt_xt) # This one looks sketchy
        f_i = true_f[i]
        midmatrix = - np.matmul(np.matmul(Kx_inverse, Kx_differentiated_wrt_xt), Kx_inverse) 
        f_prior_term[t] = - 0.5 * np.dot(np.dot(f_i, midmatrix), f_i.T)
        ################
        logdet_term[t] = - N/2 * np.trace(np.matmul(Kx_inverse, Kx_differentiated_wrt_xt))

    # x prior term ######
    ####################
    x_prior_term = - np.dot(K_t_inverse, X_estimate)

#    print("fpriorterm\n", f_prior_term)
#    print("logdetterm\n", logdet_term)
#    print("xpriorterm\n", x_prior_term)
    
    jacobian = f_prior_term + logdet_term + x_prior_term
    return - jacobian

########################
# Covariance functions #
########################
def make_Kx_with_sigma(T, X_estimate):
    Kx_fit_at_observations = np.zeros((T,T))
    for x1 in range(T):
        for x2 in range(T):
            Kx_fit_at_observations[x1,x2] = squared_exponential_covariance(X_estimate[x1],X_estimate[x2], sigma_f_fit, delta_f_fit)
    Kx_fit_at_observations = Kx_fit_at_observations  + np.identity(T)*sigma_n
    return Kx_fit_at_observations

K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)
K_t_squareroot = scipy.linalg.sqrtm(K_t)
K_t_inverse_squareroot = scipy.linalg.sqrtm(K_t_inverse)

# Set initial X
X_initial = np.linspace(np.pi/4,3/2*np.pi,T)
#np.linspace(np.pi/4,3/2*np.pi,T) 
#np.repeat(np.pi,T) 
#path + 0.6*np.sin(np.linspace(0,2*np.pi,T)) 
#2*np.pi*np.random.random(T)
print("initial\n",simplified_x_posterior(X_initial))
X_estimate = X_initial # path + 0.6*np.sin(np.linspace(0,2*np.pi,T)) #np.random.random((N,T)) #path + 0.3*np.sin(np.linspace(0,2*np.pi,T))

print("Test L value for different values of X")
print("path\n",simplified_x_posterior(path))
print("path + 0.1*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 0.1*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.2*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 0.2*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.3*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 0.3*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.4*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 0.4*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.5*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 0.5*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.6*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 0.6*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 0.7*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 0.7*np.sin(np.linspace(0,2*np.pi,T))))
print("path + 1.0*np.sin(np.linspace(0,2*np.pi,T))\n",simplified_x_posterior(path + 1.0*np.sin(np.linspace(0,2*np.pi,T))))
print("Random start\n",simplified_x_posterior(2*np.pi*np.random.random(T)))

print("Test gradient at path and slightly off the path")
print("The gradient has smaller value at 'path' but is not zero")
print("path\n",x_jacobian(path))
print("path + 0.6*np.sin(np.linspace(0,2*np.pi,T))\n",x_jacobian(path + 0.6*np.sin(np.linspace(0,2*np.pi,T))))

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
    if GRADIENT==True:
        optimization_result = optimize.minimize(simplified_x_posterior, X_estimate, jac=x_jacobian, method = "L-BFGS-B", options = {'disp':True}) # jac=x_jacobian,
    elif GRADIENT == False:
        optimization_result = optimize.minimize(simplified_x_posterior, X_estimate, method = "L-BFGS-B", options = {'disp':True})
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



