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

LIKELIHOOD_MODEL = "poisson"

##################################
# Parameters for data generation #
##################################
downsampling_factor = 4
T = 2000 #T_maximum
offset = 68170 #68170 #1000 #1751

######################################
## Loading data                     ##
######################################
## 1) Load data variables
name = sys.argv[1] #'Mouse28-140313_stuff_BS0030_awakedata.mat'
mat = scipy.io.loadmat(name)
headangle = ravel(array(mat['headangle'])) # Observed head direction
cellspikes = array(mat['cellspikes']) # Observed spike time points
cellnames = array(mat['cellnames']) # Alphanumeric identifiers for cells
trackingtimes = ravel(array(mat['trackingtimes'])) # Time stamps of head direction observations
path = headangle
T_maximum = len(path)
print("T_maximum", T_maximum)
if offset + T*downsampling_factor > T_maximum:
    sys.exit("Combination of offset, downsampling and T places the end of path outside T_maximum. Choose lower T, offset or downsampling factor.")
#print(cellspikes)
#print("Comparing length of data:")
#print("shape(path)", shape(path))
#print("shape(cellspikes)", shape(cellspikes))
#print("shape(cellnames)", shape(cellnames))
#print("shape(trackingtimes)", shape(trackingtimes))
#for i in range(len(cellspikes[0])):
#    print(shape(cellspikes[0,i]))
#print("path[0:10]",path[0:10])
#print("trackingtimes[0:10]",trackingtimes[0:10])
#print("(trackingtimes[1:]-trackingtimes[:(-1)])[0:10]",(trackingtimes[1:]-trackingtimes[:(-1)])[0:10])
#print(max(trackingtimes[1:]-trackingtimes[:(-1)]))
#print(min(trackingtimes[1:]-trackingtimes[:(-1)]))

## 1) Remove headangles where the headangle value is NaN
# Spikes for Nan values are removed in step 2)
print("How many NaN elements in path:", sum(np.isnan(path)))
whiches = np.isnan(path)
#plt.figure()
#plt.plot(whiches*1)
#plt.show()
path = path[~whiches]

## 1.5) Make path continuous where it moves from 0 to 2pi
for t in range(1,len(path)):
    if (path[t] - path[t-1]) < - np.pi:
        path[t:] += 2*np.pi
    if (path[t] - path[t-1]) > np.pi:
        path[t:] -= 2*np.pi

## 2) Since spikes are recorded as time points, we must make a matrix with counts 0,1,2,3,4
# Here we also remove spikes that happen at NaN headangles, and then we downsample the spike matrix by summing over bins
starttime = min(trackingtimes)
tracking_interval = mean(trackingtimes[1:]-trackingtimes[:(-1)])
print("Observation frequency for path, and binsize for initial sampling:", tracking_interval)
binsize = tracking_interval
nbins = len(trackingtimes)
print("Number of bins:", nbins)
print("Putting spikes in bins and making a matrix of it...")
binnedspikes = zeros((len(cellnames), nbins))
for i in range(len(cellnames)):
    spikes = ravel((cellspikes[0])[i])
    for j in range(len(spikes)):
        # note 1ms binning means that number of ms from start is the correct index
        timebin = int(floor(  (spikes[j] - starttime)/float(binsize)  ))
        if(timebin>nbins-1 or timebin<0): # check if outside bounds of the awake time
            continue
        binnedspikes[i,timebin] += 1 # add a spike to the thing
originalbinnedspikes = np.copy(binnedspikes)

# Now remove spikes for NaN path values
binnedspikes = binnedspikes[:,~whiches]
# And downsample
binsize = downsampling_factor * tracking_interval
nbins = len(trackingtimes) // downsampling_factor
print("New bin size after downsampling:", binsize)
print("Number of bins:", nbins)
downsampled_binnedspikes = np.zeros((len(cellnames), nbins))
for i in range(len(cellnames)):
    for j in range(nbins):
        downsampled_binnedspikes[i,j] = sum(binnedspikes[i,downsampling_factor*j:downsampling_factor*(j+1)])
binnedspikes = downsampled_binnedspikes

if LIKELIHOOD_MODEL == "bernoulli":
    binnedspikes = (binnedspikes>0)*1

## 3) Select an interval of time and deal with downsampling
# We need to downsample the observed head direction when we tamper with the binsize (Here we chop off the end of the observations)
downsampled_path = np.zeros(len(path) // downsampling_factor)
for i in range(len(path) // downsampling_factor):
    downsampled_path[i] = mean(path[downsampling_factor*i:downsampling_factor*(i+1)])
path = downsampled_path
# Then do downsampled offset
downsampled_offset = offset // downsampling_factor
path = path[downsampled_offset:downsampled_offset+T]
binnedspikes = binnedspikes[:,downsampled_offset:downsampled_offset+T]
# From here we really only need path (1 by T) and binnedspikes (N by T)
#T_maximum = len(path) // downsampling_factor
#print("T_maximum after downsampling", T_maximum)

## plot head direction for the selected interval
plt.figure(figsize=(10,2))
plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
#plt.plot(trackingtimes, path, '.', color='black', markersize=1.) # trackingtimes as x optional
#plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
plt.xlabel("Time")
plt.ylabel("x")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-new-data-treatment-headdirection.pdf",format="pdf")

# Plot binned spikes for the selected interval (Bernoulli style since they are binned)
bernoullispikes = (binnedspikes>0)*1
plt.figure(figsize=(10,5))
for i in range(len(cellnames)):
    plt.plot(bernoullispikes[i,:]*(i+1), '|', color='black', markersize=1.)
    plt.ylabel("neuron")
    plt.xlabel("time")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-new-data-treatment-binnedspikes.png",format="png")

## Find out which neurons are tuned to the head direction (in our interval)
number_of_X_bins = 50
bins = np.linspace(min(path)-0.000001, max(path) + 0.0000001, num=number_of_X_bins + 1)
x_grid = 0.5*(bins[:(-1)]+bins[1:])
# Poisson: Find average observed spike count
print("Find average observed spike count")
observed_spike_rate = zeros((len(cellnames), number_of_X_bins))
for i in range(len(cellnames)):
    for x in range(number_of_X_bins):
        timesinbin = (path>bins[x])*(path<bins[x+1])
        if(sum(timesinbin)>0):
            observed_spike_rate[i,x] = mean(binnedspikes[i, timesinbin] )
# Plot observed spike rates
for n4 in range(len(cellnames)//4):
    plt.figure(figsize=(10,8))
    neuron = np.array([[0,1],[2,3]])
    neuron = neuron + 4*n4
    for i in range(2):
        for j in range(2):
            plt.subplot(2,2,i*2+j+1)
            plt.plot(x_grid, observed_spike_rate[neuron[i,j],:], color=plt.cm.viridis(0.1))
            plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_spike_rate[neuron[i,j],:])))
            plt.xlabel("X")
            plt.ylabel("Average number of spikes")
            plt.title("Neuron "+str(neuron[i,j])+" with "+str(sum(binnedspikes[neuron[i,j],:]))+" spikes")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-new-data-treatment-tuning"+str(n4+1)+".png")
for i in range(4*(len(cellnames)//4), len(cellnames)):
    plt.figure()
    plt.plot(x_grid, observed_spike_rate[i,:], color=plt.cm.viridis(0.1))
    plt.title("Neuron "+str(i)+" with "+str(sum(binnedspikes[i,:]))+" spikes")
    plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_spike_rate[i,:])))
    plt.xlabel("X")
    plt.ylabel("Average number of spikes")
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-new-data-treatment-tuning-"+str(i)+".png")
plt.show()

## 5) Remove neurons that are not actually tuned to head direction
# On the entire range of time, these neurons are tuned to head direction
neuronsthataretunedtoheaddirection = [17,18,20,21,22,23,24,25,26,27,28,29,31,32,34,35,36,37,38,39,68]
# But for our current selected T and offset, we want these only: 
sgood = np.zeros(len(cellnames))<1 
for i in range(len(cellnames)):
    if i not in neuronsthataretunedtoheaddirection:
        sgood[i] = False
binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]
print("How many neurons are tuned to head direction:",len(cellnames))

## 6) Choose best offset based on where these neurons spike the most:
spike_count_in_time = np.sum(originalbinnedspikes, axis=0)
plt.figure()
plt.plot(spike_count_in_time)
plt.xlabel("Time")
plt.title("Average number of spikes for chosen neurons")
plt.show()

## 7) Change names to fit the rest of the code
N = len(cellnames) #51 with cutoff at 1000 spikes
print("N:",N)
y_spikes = binnedspikes
print("mean(y_spikes)",mean(y_spikes))
print("mean(y_spikes>0)",mean(y_spikes[y_spikes>0]))
# Spike distribution evaluation
spike_count = np.ndarray.flatten(binnedspikes)
print("How many times are there more than one spike:", sum(spike_count>1))
print("Percentage of those bins with 1 that actually have more than 1 spike:", sum(spike_count>1) / sum(spike_count>0)) #len(spike_count[spike_count>0]))
# Remove zero entries:
#spike_count = spike_count[spike_count>0]
plt.figure()
plt.hist(spike_count, bins=np.arange(0,int(max(spike_count))+1)-0.5, log=True, color=plt.cm.viridis(0.3))
plt.ylabel("Number of bins")
plt.xlabel("Spike count")
plt.title("Spike histogram")
plt.xticks(range(0,int(max(spike_count)),1))
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-spike-histogram-log.png")

# Plot y spikes
fig, ax = plt.subplots()
foo_mat = ax.matshow(y_spikes) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("y spikes")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-em-y-spikes.png")
