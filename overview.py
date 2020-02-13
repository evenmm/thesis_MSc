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
numpy.random.seed(11)

# Visualization of the GPLVM: 1) tuning curve 2) latent path 3) spikes

## 1 Draw a path randomly from its prior

## 1 Define a grid of points in X direction - draw a tuning curve on this grid from the prior
# The points of the path will not be exactly on this grid.
# But we make a MAP estimate of the tuning curve evaluated on the path.

## 3 Use this f estimate to sample spikes

def exponential_covariance(t1,t2):
    distance = abs(t1-t2)
    return r_parameter * exp(-distance/l_parameter)

def gaussian_periodic_covariance(x1,x2):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return sigma * exp(-distancesquared/(2*delta))

def gaussian_NONPERIODIC_covariance(x1,x2):
    distancesquared = (x1-x2)**2
    return sigma * exp(-distancesquared/(2*delta))

# Model parameters: 
X_dim = 40 # Number of grid points
sigma = 4 # Variance for Kx
delta = 4 # Scale for Kx
N = 1 # number of neurons 
sigma_epsilon = 0.05 # Uncertainty of observations
T = 2000
r_parameter = 10 # variance for kt 
l_parameter = 700 # length for kt

## 1. Generate random latent variable GP path
Kt = np.zeros((T, T)) 
for t1 in range(T):
    for t2 in range(T):
        Kt[t1,t2] = exponential_covariance(t1,t2)
Kt = Kt + np.identity(T)*10e-10 

# Plotting Kt
fig, ax = plt.subplots()
#ax.set(title='Kt')
ktmat = ax.matshow(Kt, cmap=plt.cm.Blues) #plt.cm.viridis
fig.colorbar(ktmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kt.pdf",format="pdf")

path = numpy.random.multivariate_normal(np.zeros(T), Kt)
#path = np.linspace(0,2*np.pi,2000)
## plot path
plt.figure()#(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=1.)
plt.xlabel("Time")
plt.ylabel("x value")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-path.pdf",format="pdf")

## 2. Define a grid of points in X direction - draw a tuning curve on this grid
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
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kx_grid.png")
# Tuning curves from GP prior
# Draw tuning curves with zero mean and covariance prior
f_tuning_curve = np.zeros((N,X_dim))
for i in range(N):
    f_tuning_curve[i] = np.random.multivariate_normal(np.zeros(X_dim),Kx_grid)
"""
Sine tuning curve
f_tuning_curve = np.array([np.sin(x_grid)])
"""
h_tuning_curve = np.exp(f_tuning_curve)

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
# Plot h
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(x_grid, h_tuning_curve[i,:], color=colors[i])
plt.xlabel("x")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-tuning-h.png")
# Plot f realizations with 95 % confidence interval
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(x_grid, np.zeros_like(x_grid), color="grey")
    plt.plot(x_grid, sigma * 1.96 * np.ones_like(x_grid), "--", color="grey")
    plt.plot(x_grid, -sigma * 1.96 * np.ones_like(x_grid), "--", color="grey")
    plt.plot(x_grid, f_tuning_curve[i,:], "-", color=colors[i])
    plt.plot(x_grid, f_tuning_curve[i,:], ".", color=colors[i])
plt.xlabel("x")
#plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-tuning-f.png")
plt.show()

N_observations = T
x_values_observed = path
f_values_observed = "HERE we must actually sample the things, NOT on the grid boy. 

"""
## Bad idea:
## Here, observations are grid points since this is the "true tuning curve"
## And "grid" could be renamed "prediction". Here this is path
## Observations and posterior, noise free
N_observations = X_dim
x_values_observed = x_grid
f_values_observed = f_tuning_curve[0]
X_dim = T
x_grid = path

Kx_grid = np.zeros((X_dim,X_dim))
for x1 in range(X_dim):
    for x2 in range(X_dim):
        Kx_grid[x1,x2] = gaussian_NONPERIODIC_covariance(x_grid[x1],x_grid[x2])
Kx_grid = Kx_grid + np.identity(X_dim)*10e-5 # To be able to invert Kx we add a small amount on the diagonal
Kx_grid_inverse = np.linalg.inv(Kx_grid)
fig, ax = plt.subplots()
kxmat = ax.matshow(Kx_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kx_path.png")
"""

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
# Plot observed data points
plt.figure()
plt.xlim(0,2*np.pi)
# Plot posterior mean
plt.plot(x_grid, mu_posterior, "-", color="grey")
# Calculate standard deviations and add 95 % confidence interval to plot
sigma_posterior = (Kx_grid) - np.dot(Kx_crossover_T, np.dot(Kx_observed_inverse, Kx_crossover))
plt.plot(x_grid, mu_posterior + 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color="grey")
plt.plot(x_grid, mu_posterior - 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color="grey")
plt.plot(x_grid, f_tuning_curve[0,:], "-", color=colors[0])
plt.plot(x_values_observed, f_values_observed, ".", color=colors[0])
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-posterior.png")

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
plt.plot(x_values_observed, f_values_observed, ".", color=colors[0])
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-noisy-posterior.png")
plt.show()

## 3. Sample spikes
# noisy_mu_posterior is our value of tuning curves 
h_tuning_curve = exp(noisy_mu_posterior)

h_spike_rate = exp(f_tuning_curve)
# Plot firing rates stupidly
colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(h_spike_rate[i], color=colors[i])
plt.xlabel("Time")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-firing-rate.pdf",format="pdf")

# Estimated tuning curves
for i in range(N):
    for x in range(number_of_X_bins):
        timesinbin = (path>bins[x])*(path<bins[x+1])
        if(sum(timesinbin)>0):
            estimated_tuning[i,x] = mean( exp(f_tuning_curve[i, timesinbin]))
        else:
            print("No observations of X between",bins[x],"and",bins[x+1],".")
# Plot tuning curves for chosen neurons

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(evaluationpoints, estimated_tuning[i,:], color=colors[i])
plt.xlabel("x value")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-fitted-tuning.pdf",format="pdf")

## 3. Generate spike trains 
# Now we actually use the __spike rates__ from hi, and not the tuning curves, to generate spikes.
y_spikes = np.zeros((N,T))
for i in range(N):
    y_spikes[i] = np.random.poisson(h_spike_rate[i])

plt.figure()
timepoints = np.linspace(0, T-1, T, endpoint=False)
# Chop off a bit for better visualization
lo_lim = 900
cutwidth = 250
timepoints = timepoints[lo_lim:lo_lim+cutwidth]
y_spikes = y_spikes[:,lo_lim:lo_lim+cutwidth]
# For each neurons, select only those times with one or more spikes
for i in range(N):
    actually_spike_times = timepoints[y_spikes[i]>0]
    actually_spike_numbers = y_spikes[i][y_spikes[i]>0]
    for j in range(len(actually_spike_times)):
        plt.plot(actually_spike_times[j], (actually_spike_numbers[j]>0)*(i+1), '|', color=colors[i], markersize=5.*actually_spike_numbers[j])
plt.ylabel("Neuron")
plt.xlabel("Time")
plt.yticks(range(1,N+1))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-spikes.pdf",format="pdf")


plt.show()


#############################
## Bernoulli
"""
h_spike_rate = exp(f_tuning_curve)/(1+exp(f_tuning_curve))
# Plot firing rates stupidly
colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(h_spike_rate[i], color=colors[i])
plt.xlabel("Time")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-firing-rate.pdf",format="pdf")

# Estimated tuning curves
for i in range(N):
    for x in range(number_of_X_bins):
        timesinbin = (path>bins[x])*(path<bins[x+1])
        if(sum(timesinbin)>0):
            estimated_tuning[i,x] = mean( exp(f_tuning_curve[i, timesinbin])/(1+exp(f_tuning_curve[i, timesinbin]))) # inverse Logit mapping from f to h
        else:
            print("No observations of X between",bins[x],"and",bins[x+1],".")
# Plot tuning curves for chosen neurons

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(evaluationpoints, estimated_tuning[i,:], color=colors[i])
plt.xlabel("X value")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-fitted-tuning.pdf",format="pdf")

## 3. Generate spike trains 
# Now we actually use the __spike rates__ from hi, and not the tuning curves, to generate spikes.
y_spikes = np.zeros((N,T))
for i in range(N):
    y_spikes[i] = np.random.poisson(h_spike_rate[i])

plt.figure()
timepoints = np.linspace(0, T-1, T, endpoint=False)
# Chop off a bit for better visualization
lo_lim = 900
cutwidth = 250
timepoints = timepoints[lo_lim:lo_lim+cutwidth]
y_spikes = y_spikes[:,lo_lim:lo_lim+cutwidth]
# For each neurons, select only those times with one or more spikes
for i in range(N):
    actually_spike_times = timepoints[y_spikes[i]>0]
    actually_spike_numbers = y_spikes[i][y_spikes[i]>0]
    for j in range(len(actually_spike_times)):
        plt.plot(actually_spike_times[j], (actually_spike_numbers[j]>0)*(i+1), '|', color=colors[i], markersize=10.*actually_spike_numbers[j])
plt.ylabel("Neuron")
plt.xlabel("Time")
plt.yticks(range(1,N+1))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-spikes.pdf",format="pdf")


plt.show()
"""