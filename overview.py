from scipy import *
import scipy.io
import scipy.ndimage
import numpy as np
import scipy.optimize as sp
import numpy.random
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
import time
import sys
plt.rc('image', cmap='viridis')
numpy.random.seed(16)

# Visualization of the GPLVM: 1) tuning curve 2) latent path 3) spikes

## 1 Draw a path randomly from its prior

## 2 Draw f values for N neurons at path values

## 2.5 To plot f, define a grid of points in X direction.
# The points of the path will not be exactly on this grid.
# But we make a MAP estimate on the grid given path values

## 3 Use f values at path to sample spikes

def exponential_covariance(t1,t2, sigma, delta):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def gaussian_periodic_covariance(x1,x2, sigma, delta):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return sigma * exp(-distancesquared/(2*delta))

def gaussian_NONPERIODIC_covariance(x1,x2, sigma, delta):
    distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

# Model parameters: 
N = 4 # number of neurons 
T = 500 # time bins
sigma_path = 10 # variance for kt 
delta_path = 700 # scale for kt
# Observed GP parameters
N_observations = T
sigma_observed = 1 # Variance for Kx
delta_observed = 1 # Scale for Kx
sigma_epsilon_observed = 0.05 # Variance of observations. Remember: Even with exact observations, the generative GP tuning curve will look noisy since it is sampled so tightly. 
# Fitting GP parameters
X_dim = 40 # Number of grid points
sigma_fit = 1.2 # Variance for Kx
delta_fit = 0.5 # Scale for Kx
sigma_epsilon_fit = 0.1 # Variance of observations

## 1. Generate random latent variable GP path
print("Making Kt and path")
Kt = np.zeros((T, T)) 
for t1 in range(T):
    for t2 in range(T):
        Kt[t1,t2] = exponential_covariance(t1,t2, sigma_path, delta_path)
Kt = Kt + np.identity(T)*10e-10

# Plotting Kt
fig, ax = plt.subplots()
ktmat = ax.matshow(Kt, cmap=plt.cm.Blues) #plt.cm.viridis
fig.colorbar(ktmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kt.pdf",format="pdf")

path = numpy.random.multivariate_normal(np.zeros(T), Kt)
path = np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi
# plot path
plt.figure()#(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=4.)
plt.xlabel("Time")
plt.ylabel("x value")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-path.png")

## 2 Sample f values for N neurons at path values
N_observations = T
x_values_observed = path
# Calculate covariance matrix Kx observed
print("Making spatial covariance matrice: Kx observed")
Kx_observed = np.zeros((N_observations,N_observations))
for x1 in range(N_observations):
    for x2 in range(N_observations):
        Kx_observed[x1,x2] = gaussian_periodic_covariance(x_values_observed[x1],x_values_observed[x2], sigma_observed, delta_observed)
# Sample exact f values
f_values_exact = np.zeros((N,T))
numpy.random.seed(4)
for i in range(N):
    f_values_exact[i] = np.random.multivariate_normal(np.zeros(T),Kx_observed)

# We can choose to add noise to the observations:
Kx_observed = Kx_observed + np.identity(N_observations)*sigma_epsilon_observed
fig, ax = plt.subplots()
kx_obs_mat = ax.matshow(Kx_observed, cmap=plt.cm.Blues)
fig.colorbar(kx_obs_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-Kx_observed.png")

# Sample f values
f_values_observed = np.zeros((N,T))
numpy.random.seed(4)
for i in range(N):
    f_values_observed[i] = np.random.multivariate_normal(np.zeros(T),Kx_observed)

## Now we want to infer grid f based on observed values.
x_grid = np.linspace(0, 2*np.pi, num=X_dim)
print("Making spatial covariance matrice: Kx_fit at observations")
Kx_fit_at_observations = np.zeros((N_observations,N_observations))
for x1 in range(N_observations):
    for x2 in range(N_observations):
        Kx_fit_at_observations[x1,x2] = gaussian_periodic_covariance(x_values_observed[x1],x_values_observed[x2], sigma_fit, delta_fit)
# By adding sigma_epsilon on the diagonal, we assume noise and make the covariance matrix positive semidefinite
Kx_fit_at_observations = Kx_fit_at_observations  + np.identity(N_observations)*sigma_epsilon_fit
Kx_fit_at_observations_inverse = np.linalg.inv(Kx_fit_at_observations)
fig, ax = plt.subplots()
kx_obs_mat = ax.matshow(Kx_fit_at_observations, cmap=plt.cm.Blues)
fig.colorbar(kx_obs_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-Kx_fit_at_observations.png")
print("Making spatial covariance matrice: Kx crossover")
Kx_crossover = np.zeros((N_observations,X_dim))
for x1 in range(N_observations):
    for x2 in range(X_dim):
        Kx_crossover[x1,x2] = gaussian_periodic_covariance(x_values_observed[x1],x_grid[x2], sigma_fit, delta_fit)
fig, ax = plt.subplots()
kx_cross_mat = ax.matshow(Kx_crossover, cmap=plt.cm.Blues)
fig.colorbar(kx_cross_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kx_crossover.png")
Kx_crossover_T = np.transpose(Kx_crossover)
print("Making spatial covariance matrice: Kx grid")
Kx_grid = np.zeros((X_dim,X_dim))
for x1 in range(X_dim):
    for x2 in range(X_dim):
        Kx_grid[x1,x2] = gaussian_periodic_covariance(x_grid[x1],x_grid[x2], sigma_fit, delta_fit)
fig, ax = plt.subplots()
kxmat = ax.matshow(Kx_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kx_grid.png")

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
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-sigma_posterior.png")

# Plot posterior mean with 95 % confidence interval
colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()
plt.xlim(0,2*np.pi)
for i in range(N):
    plt.plot(x_grid, mu_posterior[i], "-", color=colors[i])
    plt.plot(x_grid, mu_posterior[i] + 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color=colors[i])
    plt.plot(x_grid, mu_posterior[i] - 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color=colors[i])
    plt.plot(x_values_observed, f_values_observed[i], ".", color="grey", markersize=2)
    #plt.plot(x_values_observed, f_values_exact[i], ".", markersize=3, color=plt.cm.viridis(0.6))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-posterior.png")

# The fitted tuning curves are nice examples
plt.figure()
plt.xlim(0,2*np.pi)
for i in range(N):
    plt.plot(x_grid, mu_posterior[i], "-", color=colors[i])
plt.xlabel("x")
plt.ylabel("Log spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-fitted-gp.png")

# Plot h
h_tuning_curve = np.exp(mu_posterior)
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(x_grid, h_tuning_curve[i,:], color=colors[i])
plt.xlabel("x")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-tuning-h.png")

## 3. Sample spikes
# Find mean prediction of log tuning on path values
pre = np.zeros((N,X_dim))
mu_posterior_on_path = np.zeros((N, T))
Kx_grid = Kx_grid + np.identity(X_dim)*sigma_fit
Kx_grid_inverse = np.linalg.inv(Kx_grid)
for i in range(N):
    pre[i] = np.dot(Kx_grid_inverse, mu_posterior[i])
    mu_posterior_on_path[i] = np.dot(Kx_crossover, pre[i])

# Exponentiate, get spike rates on path
h_spike_rate_on_path = np.exp(mu_posterior_on_path)

# Sample spikes
y_spikes = np.zeros((N,T))
for i in range(N):
    y_spikes[i] = np.random.poisson(h_spike_rate_on_path[i])

# Plot spikes
plt.figure()
timepoints = np.linspace(0, T, T, endpoint=False)
for i in range(N):
    for j in range(T):
        plt.plot(timepoints[j], (i+1) , '|', color=colors[i], markersize=5.*y_spikes[i][j])
plt.ylabel("Neuron")
plt.xlabel("Time")
plt.yticks(range(1,N+1))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-spikes.png")
plt.tight_layout()

# Ultimate plot
# Plot rotated tuning curves
from matplotlib import gridspec
fig = plt.figure(figsize=(7,4))
gs  = gridspec.GridSpec(1, 2, width_ratios=[0.7, 4])
plt.subplot(gs[0])
base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(90)
for i in range(N):
    plt.plot(x_grid, 1 + h_tuning_curve[i,:], color=colors[i], transform= rot + base)
    #line = plt.plot(evaluationpoints, estimated_tuning[plotcurves[i],:], color=colors[i], transform= rot + base)
#frame1=plt.gca()
#frame1.axes.get_xaxis().set_visible(False)
#frame1.axes.get_yaxis().set_visible(False)
plt.axis("off")
plt.subplot(gs[1])

# Plot spikes and latent variable
for i in range(N):
    plt.plot(timepoints, path, '.', color='black', markersize=3)
    nonzero = y_spikes[i,:]>0
    plt.xlabel("Time")
    plt.ylabel("X value")
    plt.ylim(0,2*np.pi)
    peak = x_grid[np.argmax(h_tuning_curve[i])]
    print(peak)
    for j in range(sum(nonzero)):
        plt.plot(timepoints[nonzero][j], (y_spikes[i][nonzero][j]>0)*peak, '|', color=colors[i], markersize=2*y_spikes[i][nonzero][j])
#    plt.xticks([])

plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-ultimate.png") #, transparent=True
plt.show()
