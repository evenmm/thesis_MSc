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
numpy.random.seed(20)

## There are two threads to follow here: 
# Visualization of the GPLVM: 1) tuning curve 2) latent path 3) spikes
# Visualization of GP inference: 1) prior 2) observations 3) posterior

## 1 Define a grid of points in X direction - draw a tuning curve on this grid
# This tuning curve is realized entirely randomly from its prior, evaluated at some points. 
# This gives a good illustration of the prior. *As* good as if we let some X path define the points

## 2 Draw a path randomly from its prior
# The points of this path will not be exactly on the grid defined in 1. 
# Therefore the value of the tuning curve GP is a random variable with a normal conditional distribution
# We use the mean of its distribution to sample spikes
# When we go the other way, from path X to grid to plot tuning curve, we will also present the mean of the distribution 

## 3 Compute spikes


def exponential_covariance(t1,t2):
    distance = abs(t1-t2)
    return r_parameter * exp(-distance/l_parameter)

def gaussian_periodic_covariance(x1,x2):
    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    return rho * exp(-distancesquared/(2*delta))

def gaussian_NONPERIODIC_covariance(x1,x2):
    distancesquared = (x1-x2)**2
    return rho * exp(-distancesquared/(2*delta))


## 1 Define a grid of points in X direction - draw a tuning curve on this grid

# Model parameters: 
X_dim = 50
rho = 1.2 # Variance
delta = 0.4 # Scale
N = 3 # number of neurons 

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
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-grid-overview-kx_grid.png")
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
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-grid-tuning.pdf",format="pdf")
# Plot f realizations with 95 % confidence interval
plt.figure()#(figsize=(10,8))
for i in range(N):
    plt.plot(x_grid, np.zeros_like(x_grid), color="grey")
    plt.plot(x_grid, rho * 1.96 * np.ones_like(x_grid), "--", color="grey")
    plt.plot(x_grid, -rho * 1.96 * np.ones_like(x_grid), "--", color="grey")
    plt.plot(x_grid, f_tuning_curve[i,:], "-", color=colors[i])
    plt.plot(x_grid, f_tuning_curve[i,:], ".", color=colors[i])
plt.xlabel("x")
#plt.ylabel("Spike rate")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-grid-tuning.pdf",format="pdf")
plt.show()

# Observations and posterior
N_observations = 4
x_array_positions = np.random.randint(0, X_dim, size=N_observations)
print(x_array_positions)
x_values_observed = x_grid[x_array_positions]
print(x_values_observed)
f_values_observed = f_tuning_curve[0][x_array_positions]
print(f_values_observed)
# Plot observed data points
plt.figure()
plt.xlim(0,2*np.pi)
plt.ylim(-rho * 1.96, rho * 1.96)

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
# Plot posterior mean
plt.plot(x_grid, f_tuning_curve[0,:], "-", color=colors[0])
plt.plot(x_grid, mu_posterior, "-", color="grey")
# Calculate standard deviations and add 95 % confidence interval to plot
sigma_posterior = np.diag(Kx_grid) - np.dot(Kx_crossover_T, np.dot(Kx_observed_inverse, Kx_crossover))
plt.plot(x_grid, mu_posterior + 1.96*np.diag(sigma_posterior), "--", color="grey")
plt.plot(x_grid, mu_posterior - 1.96*np.diag(sigma_posterior), "--", color="grey")
plt.plot(x_values_observed, f_values_observed, ".", color=colors[0])
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-grid-overview-posterior.png")
plt.show()
"""
# Checking if correct by pytting the tuning at every time equal to X, giving spike prob of 1 over entire interval
#f_tuning_curve[-1] = [100 for i in range(len(f_tuning_curve[-1]))] # approved

# Extracting actual tuning curve from the simulated Gaussian process f
# Now doing Bernoulli spiking to see. For Poisson spiking:
# h_tuning_curve = np.exp(f_tuning_curve)

number_of_X_bins = 100
bins = np.linspace(min(path)-0.000001, max(path)+0.0000001, num=number_of_X_bins + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])
estimated_tuning = np.zeros((N, number_of_X_bins))

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
"""

############################################################################################################
## 2. Generate random latent variable GP path
T = 2000
# Create Kt
r_parameter = 10
l_parameter = 700

Kt = np.zeros((T, T)) 
for t1 in range(T):
    for t2 in range(T):
        Kt[t1,t2] = exponential_covariance(t1,t2)
Kt = Kt + np.identity(T)*10e-10 # To be able to invert Kt we add a small amount on the diagonal

# Plotting Kt
fig, ax = plt.subplots()
#ax.set(title='Kt')
ktmat = ax.matshow(Kt, cmap=plt.cm.Blues) #plt.cm.viridis
fig.colorbar(ktmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-kt.pdf",format="pdf")

path = numpy.random.multivariate_normal(np.zeros(T), Kt)
#path = np.linspace(0,2*np.pi,2000)
## plot path
plt.figure()#(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=1.)
plt.xlabel("Time")
plt.ylabel("x value")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-path.pdf",format="pdf")

## 2. Generate tuning curves
N = 3 # number of neurons 

# Create Kx
rho = 0.5 
delta = 1
Kx = np.zeros((T, T)) 
for t1 in range(T):
    for t2 in range(T):
        Kx[t1,t2] = gaussian_periodic_covariance(path[t1],path[t2])
Kx = Kx + np.identity(T)*10e-5 # To be able to invert Kx we add a small amount on the diagonal

# Plotting Kx
fig, ax = plt.subplots()
#ax.set(title='Kx')
kxmat = ax.matshow(Kx, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"new-overview-kx.pdf",format="pdf")

f_tuning_curve = np.zeros((N,T))
for i in range(N):
    f_tuning_curve[i] = np.random.multivariate_normal(np.zeros(T),Kx)

# Checking if correct by pytting the tuning at every time equal to X, giving spike prob of 1 over entire interval
#f_tuning_curve[-1] = [100 for i in range(len(f_tuning_curve[-1]))] # approved

# Extracting actual tuning curve from the simulated Gaussian process f
# Now doing Bernoulli spiking to see. For Poisson spiking:
# h_tuning_curve = np.exp(f_tuning_curve)

number_of_X_bins = 100
bins = np.linspace(min(path)-0.000001, max(path)+0.0000001, num=number_of_X_bins + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])
estimated_tuning = np.zeros((N, number_of_X_bins))

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