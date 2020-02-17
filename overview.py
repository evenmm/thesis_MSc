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
N = 1 # number of neurons 
T = 1000 # time bins
r_parameter = 10 # variance for kt 
l_parameter = 700 # scale for kt
# Generative GP
sigma = 1.2 # Variance for Kx
delta = 1 # Scale for Kx
sigma_epsilon = 0.05 # Variance of observations
# Inferring GP on grid
X_dim = 40 # Number of grid points

## 1. Generate random latent variable GP path
print("Making Kt")
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
# Truncate to keep it between 0 and 2pi
path = np.mod(path, 2*np.pi)
    # plot path
plt.figure()#(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=1.)
plt.xlabel("Time")
plt.ylabel("x value")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-path.pdf",format="pdf")

## 2 Sample f values for N neurons at path values
print("Making path")
N_observations = T
x_values_observed = path
# Calculate covariance matrix Kx observed
print("Making spatial covariance matrice: Kx observed")
Kx_observed = np.zeros((N_observations,N_observations))
for x1 in range(N_observations):
    for x2 in range(N_observations):
        Kx_observed[x1,x2] = gaussian_periodic_covariance(x_values_observed[x1],x_values_observed[x2])
# since we have noisy observations, we must add sigma_epsilon on the diagonal to make the covariance matrix positive semidefinite
Kx_observed = Kx_observed  + np.identity(N_observations)*sigma_epsilon
fig, ax = plt.subplots()
kx_obs_mat = ax.matshow(Kx_observed, cmap=plt.cm.Blues)
fig.colorbar(kx_obs_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-Kx_observed.png")
Kx_observed_inverse = np.linalg.inv(Kx_observed)


## Now we want to infer f based on observed values.
# Sample f values
f_values_observed = np.zeros((N,T))
for i in range(N):
    f_values_observed[i] = np.random.multivariate_normal(np.zeros(T),Kx_observed)
## 2.5 To plot tuning curve
## Define a grid of points in X direction - infer mean on the grid
x_grid = np.linspace(0, 2*np.pi, num=X_dim)

print("Making spatial covariance matrice: Kx crossover")
Kx_crossover = np.zeros((N_observations,X_dim))
for x1 in range(N_observations):
    for x2 in range(X_dim):
        Kx_crossover[x1,x2] = gaussian_periodic_covariance(x_values_observed[x1],x_grid[x2])
fig, ax = plt.subplots()
kx_cross_mat = ax.matshow(Kx_crossover, cmap=plt.cm.Blues)
fig.colorbar(kx_cross_mat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kx_crossover.png")
Kx_crossover_T = np.transpose(Kx_crossover)
print("Making spatial covariance matrice: Kx grid")
Kx_grid = np.zeros((X_dim,X_dim))
for x1 in range(X_dim):
    for x2 in range(X_dim):
        Kx_grid[x1,x2] = gaussian_periodic_covariance(x_grid[x1],x_grid[x2])
fig, ax = plt.subplots()
kxmat = ax.matshow(Kx_grid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-kx_grid.png")

# Infer mean on the grid
pre = np.zeros((N,T))
mu_posterior = np.zeros((N, X_dim))
for i in range(N):
    pre[i] = np.dot(Kx_observed_inverse, f_values_observed[i])
    mu_posterior[i] = np.dot(Kx_crossover_T, pre[i])
# Plot posterior mean
# Calculate standard deviations and add 95 % confidence interval to plot
colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()
plt.xlim(0,2*np.pi)
plt.plot(x_grid, mu_posterior[0], "-", color=colors[0])
sigma_posterior = (Kx_grid) - np.dot(Kx_crossover_T, np.dot(Kx_observed_inverse, Kx_crossover))
plt.plot(x_grid, mu_posterior[0] + 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color=colors[0])
plt.plot(x_grid, mu_posterior[0] - 1.96*np.sqrt(np.diag(sigma_posterior)), "--", color=colors[0])
plt.plot(x_values_observed, f_values_observed[0], ".", color="grey", markersize=2)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-posterior.png")
plt.show()

# Plot h
h_tuning_curve = np.exp(f_tuning_curve)
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(x_grid, h_tuning_curve[i,:], color=colors[i])
plt.xlabel("x")
plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-overview-tuning-h.png")
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