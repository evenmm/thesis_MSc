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
numpy.random.seed(20) #20

# Visualization of GP inference: 1) prior 2) observations 3) posterior

def exponential_covariance(t1,t2):
    distance = abs(t1-t2)
    return sigma * exp(-distance/delta)

def gaussian_periodic_covariance(x1,x2):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return sigma * exp(-distancesquared/(2*delta))

def gaussian_NONPERIODIC_covariance(x1,x2):
    distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

## Tuning curve GP
# Define a grid of points in X direction - draw a tuning curve on this grid

# Model parameters: 
X_dim = 50
sigma = 1.2 # Variance
delta = 1 # Scale 
N = 3 # number of neurons 
sigma_epsilon = 0.05
N_observations_total = 10

x_grid = np.linspace(0, 2*np.pi, num=X_dim)
Kx_grid = np.zeros((X_dim,X_dim))
for x1 in range(X_dim):
    for x2 in range(X_dim):
        Kx_grid[x1,x2] = gaussian_NONPERIODIC_covariance(x_grid[x1],x_grid[x2])
Kx_grid = Kx_grid + np.identity(X_dim)*10e-5 # To be able to invert Kx we add a small amount on the diagonal
Kx_grid_inverse = np.linalg.inv(Kx_grid)
fig, ax = plt.subplots()
kxmat = ax.matshow(Kx_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-kx_grid.png")
#plt.show()
# Draw tuning curves with zero mean and covariance prior
f_tuning_curve = np.zeros((N,X_dim))
for i in range(N):
    f_tuning_curve[i] = np.random.multivariate_normal(np.zeros(X_dim),Kx_grid)
h_tuning_curve = np.exp(f_tuning_curve)

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
# Plot h
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(x_grid, h_tuning_curve[i,:], color=colors[i])
plt.xlabel("x")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-tuning-h.png")
# Plot f realizations with 95 % confidence interval
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(x_grid, np.zeros_like(x_grid), color="grey")
    plt.plot(x_grid, sigma * 1.96 * np.ones_like(x_grid), "--", color="grey")
    plt.plot(x_grid, -sigma * 1.96 * np.ones_like(x_grid), "--", color="grey")
    plt.plot(x_grid, f_tuning_curve[i,:], "-", color=colors[i])
#    plt.plot(x_grid, f_tuning_curve[i,:], ".", color=colors[i])
plt.xlabel("x")
#plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-tuning-f.png")
# Plot prior
mu_prior = np.zeros(X_dim)
sigma_prior = sigma * np.identity(X_dim)
plt.figure()
plt.xlim(0,2*np.pi)
plt.plot(x_grid, f_tuning_curve[0,:], "-", color=colors[0])
plt.plot(x_grid, mu_prior, "-", color="grey")
plt.plot(x_grid, mu_prior + 1.96*np.sqrt(np.diag(sigma_prior)), "--", color="grey")
plt.plot(x_grid, mu_prior - 1.96*np.sqrt(np.diag(sigma_prior)), "--", color="grey")
plt.ylim(-3,3)
plt.xlabel("X")
plt.ylabel("f(X)")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-prior.png")

## Observations
x_array_positions = np.random.randint(0, X_dim, size=N_observations_total)
for N_observations in range(1,N_observations_total+1):
    x_values_observed = x_grid[x_array_positions[0:N_observations]]
    print(x_values_observed)
    f_values_observed = f_tuning_curve[0][x_array_positions[0:N_observations]]

    # Calculate covariance matrices
    Kx_observed = np.zeros((N_observations,N_observations))
    for x1 in range(N_observations):
        for x2 in range(N_observations):
            Kx_observed[x1,x2] = gaussian_NONPERIODIC_covariance(x_values_observed[x1],x_values_observed[x2])
    Kx_observed_inverse = np.linalg.inv(Kx_observed)

    Kx_crossover = np.zeros((N_observations,X_dim))
    for x1 in range(N_observations):
        for x2 in range(X_dim):
            Kx_crossover[x1,x2] = gaussian_NONPERIODIC_covariance(x_values_observed[x1],x_grid[x2])

    Kx_crossover_T = np.transpose(Kx_crossover)
    # Calculate posterior mean function
    pre = np.dot(Kx_observed_inverse, f_values_observed)
    mu_posterior = np.dot(Kx_crossover_T, pre)
    plt.figure()
    plt.xlim(0,2*np.pi)
    # Plot posterior mean
    plt.plot(x_grid, mu_posterior, "-", color="grey")
    # Calculate standard deviations and add 95 % confidence interval to plot
    sigma_posterior = (Kx_grid) - np.dot(Kx_crossover_T, np.dot(Kx_observed_inverse, Kx_crossover))
    plt.plot(x_grid, mu_posterior + 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color="grey")
    plt.plot(x_grid, mu_posterior - 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color="grey")
    plt.plot(x_grid, f_tuning_curve[0,:], "-", color=colors[0])
    plt.plot(x_values_observed, f_values_observed, ".", color=colors[0], markersize=10) # Plot observed data points
    plt.ylim(-3,3)
    plt.xlabel("X")
    plt.ylabel("f(X)")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-posterior-%s.png" % int(N_observations))

    ## Noisy observations
    noisy_Kx_observed = Kx_observed + sigma_epsilon*np.eye(N_observations)
    noisy_Kx_observed_inverse = np.linalg.inv(noisy_Kx_observed)
    noisy_pre = np.dot(noisy_Kx_observed_inverse, f_values_observed)
    noisy_mu_posterior = np.dot(Kx_crossover_T, pre)
    noisy_sigma_posterior = (Kx_grid) - np.dot(Kx_crossover_T, np.dot(noisy_Kx_observed_inverse, Kx_crossover))    
    plt.figure()
    plt.xlim(0,2*np.pi)
    plt.plot(x_grid, noisy_mu_posterior, "-", color="grey")
    plt.plot(x_grid, noisy_mu_posterior + 1.96*np.sqrt(np.diag(noisy_sigma_posterior)), "--", color="grey")
    plt.plot(x_grid, noisy_mu_posterior - 1.96*np.sqrt(np.diag(noisy_sigma_posterior)), "--", color="grey")
    plt.plot(x_grid, f_tuning_curve[0,:], "-", color=colors[0])
    plt.plot(x_values_observed, f_values_observed, ".", color=colors[0], markersize=10)
    plt.xlabel("X")
    plt.ylabel("f(X)")
    plt.ylim(-3,3)
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-noisy-posterior-%s.png" % int(N_observations))

## Latent variable GP
T = 200
sigma = 1 # Variance
delta = 50 # Scale 
sigma_epsilon = 0.0

print("Making Kt")
Kt = np.zeros((T, T)) 
for t1 in range(T):
    for t2 in range(T):
        Kt[t1,t2] = exponential_covariance(t1,t2)

# Plotting Kt
fig, ax = plt.subplots()
ktmat = ax.matshow(Kt, cmap=plt.cm.Blues) #plt.cm.viridis
fig.colorbar(ktmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-kt.pdf",format="pdf")

path = numpy.random.multivariate_normal(np.zeros(T), Kt)
#path = np.mod(path, 2*np.pi) # Truncate to keep it between 0 and 2pi
# plot path
plt.figure()#(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=1.)
plt.xlabel("Time")
plt.ylabel("x value")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-gp-overview-path.pdf",format="pdf")

plt.show()
