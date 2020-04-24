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
numpy.random.seed(1)

##############
# Parameters #
##############
T = 400
N = 100
N_iterations = 1
sigma_n = 0.4 # Assumed variance of observations for the GP that is fitted. 10e-5
lr = 0.9 # Learning rate by which we multiply sigma_n at every iteration

N_inducing_points = 70 # Number of inducing points. Wu uses 25 in 1D and 10 per dim in 2D
N_plotgridpoints = 100 # Number of grid points for plotting f posterior only 
LIKELIHOOD_MODEL = "poisson" # "bernoulli" "poisson"
COVARIANCE_KERNEL_KX = "nonperiodic" # "periodic" "nonperiodic"
TUNINGCURVE_DEFINITION = "bumps" # "triangles" "bumps"
sigma_f_fit = 2 # Variance for the tuning curve GP that is fitted. 8
delta_f_fit = 0.1 # Scale for the tuning curve GP that is fitted. 0.3
sigma_x = 6 # Variance of X for K_t
delta_x = 10 # Scale of X for K_t
P = 1 # Dimensions of latent variable 
GRADIENT_FLAG = False # Choose to use gradient or not

print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("\nUsing gradient?", GRADIENT_FLAG, "\n\n")
print("True tuning curve shape:", TUNINGCURVE_DEFINITION)

if TUNINGCURVE_DEFINITION == "triangles":
    tuningwidth = 1 # width of tuning (in radians)
    biasterm = -2 # Average H outside tuningwidth -4
    tuningcovariatestrength = np.linspace(0.5*tuningwidth,10.*tuningwidth, N) # H value at centre of tuningwidth 6*tuningwidth
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

#path = np.pi + numpy.random.multivariate_normal(np.zeros(T), Kt)
#path[40:70] = path[40] + np.sin(np.linspace(0,4*np.pi,30))
#path[70:100] = np.linspace(path[70],np.pi,30)
path = np.pi + numpy.random.multivariate_normal(np.zeros(T), Kt)
#path = 1.5 + 0.4*np.linspace(0,2*np.pi,T) + 0.7*numpy.random.multivariate_normal(np.zeros(T), Kt) #
# np.pi + np.pi*np.sin(np.linspace(0,10*np.pi,T))
# np.linspace(0,2*np.pi,T)
#np.array(np.pi + 1*np.pi*np.sin([2*np.pi*t/T for t in range(T)]))
#numpy.random.multivariate_normal(np.zeros(T), Kt)
#np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi
#path = np.linspace(0,2*np.pi,T) + 0.5*np.pi*np.sin(np.linspace(0,10*np.pi,T)) #!!!

fig, ax = plt.subplots(figsize=(2,8))
plt.title("Latent path")
ax.plot(path, np.linspace(0,T,T), color="black", label='True X')
plt.gca().invert_yaxis()
ax.xaxis.tick_top()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-path.png")
plt.tight_layout()
plt.show()

########################
# Covariance functions #
########################

# Inducing points based on where the X actually are
x_grid_induce = np.linspace(min(path), max(path), N_inducing_points) 
K_gg = np.zeros((N_inducing_points,N_inducing_points))
for x1 in range(N_inducing_points):
    for x2 in range(N_inducing_points):
        K_gg[x1,x2] = squared_exponential_covariance(x_grid_induce[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
fig, ax = plt.subplots()
foo_mat = ax.matshow(K_gg, cmap=plt.cm.Blues)
fig.colorbar(foo_mat, ax=ax)
plt.title("Kggplain")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-kgg.png")

K_gg = K_gg + sigma_n*np.identity(N_inducing_points)
K_gg_inverse = np.linalg.inv(K_gg)

fig, ax = plt.subplots()
foo_mat = ax.matshow(K_gg_inverse, cmap=plt.cm.Blues)
fig.colorbar(foo_mat, ax=ax)
plt.title("Kgg inverse")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-kgg.png")

K_t = np.zeros((T,T))
for t1 in range(T):
    for t2 in range(T):
        K_t[t1,t2] = exponential_covariance(t1,t2, sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)


K_xg = np.zeros((T,N_inducing_points))
for x1 in range(T):
    for x2 in range(N_inducing_points):
        K_xg[x1,x2] = squared_exponential_covariance(path[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
K_gx = K_xg.T
fig, ax = plt.subplots(figsize=(2,8))
foo_mat = ax.matshow(K_xg, cmap=plt.cm.Blues)
plt.title("Kxg")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-kxg.png")
fig, ax = plt.subplots(figsize=(8,2))
foo_mat = ax.matshow(K_gx, cmap=plt.cm.Blues)
plt.title("Kgx")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-kgx.png")

# Plot standard Kx and "KxgKgg-1Kgx" for comparison
K_x_plain = np.zeros((T,T))
for x1 in range(T):
    for x2 in range(T):
        K_x_plain[x1,x2] = squared_exponential_covariance(path[x1],path[x2], sigma_f_fit, delta_f_fit)
# By adding sigma_epsilon on the diagonal, we assume noise and make the covariance matrix positive semidefinite
#K_x_plain = K_x_plain  + np.identity(T)*sigma_n
fig, ax = plt.subplots()
foo_mat = ax.matshow(K_x_plain, cmap=plt.cm.Blues)
plt.title("Kx plain")
fig.colorbar(foo_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-kx-plain.png")

K_x_tilde = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx)
fig, ax = plt.subplots()
foo_mat = ax.matshow(K_x_tilde, cmap=plt.cm.Blues)
plt.title("Inducing product")
fig.colorbar(foo_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-kx-tilde.png")

K_x_difference = K_x_plain - K_x_tilde
fig, ax = plt.subplots()
foo_mat = ax.matshow(K_x_difference, cmap=plt.cm.Blues)
plt.title("difference")
fig.colorbar(foo_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-Kx-matrices-kx-tilde.png")
plt.show()
