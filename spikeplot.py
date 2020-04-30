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
T = 2000 #1500 #1000

##################################
# Parameters for data generation #
##################################
downsampling_factor = 1
offset = 68170 #1000 #1750
thresholdforneuronstokeep = 1000 # number of spikes to be considered useful

######################################
## Loading data                     ##
######################################
name = sys.argv[1] #'Mouse28-140313_stuff_BS0030_awakedata.mat'

mat = scipy.io.loadmat(name)
headangle = ravel(array(mat['headangle'])) # Observed head direction
cellspikes = array(mat['cellspikes']) # Observed spike time points
cellnames = array(mat['cellnames']) # Alphanumeric identifiers for cells
trackingtimes = ravel(array(mat['trackingtimes'])) # Time stamps of path observations

## define bin size and make matrix of spikes
startt = min(trackingtimes)
tracking_interval = mean(trackingtimes[1:]-trackingtimes[:(-1)])
print("Interval between head direction observations:", tracking_interval)
binsize = downsampling_factor * tracking_interval
print("Bin size for spike counts:", binsize)
print("Make sure that path and spikes are still aligned!!!")
# We need to downsample the observed head direction as well when we tamper with the binsize
#downsampled = np.zeros(len(headangle) // downsampling_factor)
#for i in range(len(headangle) // downsampling_factor):
#    downsampled[i] = mean(headangle[downsampling_factor*i:downsampling_factor*(i+1)])
#headangle = downsampled
print("Binning spikes...")
nbins = len(trackingtimes) // downsampling_factor
binnedspikes = zeros((len(cellnames), nbins))
sgood = zeros(len(binnedspikes[:,0]))<1
for i in range(len(cellnames)):
  spikes = ravel((cellspikes[0])[i])
  for j in range(len(spikes)):
    # note 1ms binning means that number of ms from start is the correct index
    tt = int(floor(  (spikes[j] - startt)/float(binsize)  ))
    if(tt>nbins-1 or tt<0): # check if outside bounds of the awake time
      continue
    binnedspikes[i,tt] += 1 # add a spike to the thing

  ## check if neuron has enough spikes
  if(sum(binnedspikes[i,:])<thresholdforneuronstokeep):
      sgood[i] = False
      continue
  ## remove neurons that are not tuned to head direction
  if i not in [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,38,47]:
      sgood[i] = False
      continue
print("Is good:",np.linspace(1,len(cellnames),len(cellnames))[sgood])
# Remove neurons that have less total spikes than a threshold 
# Also remove those that are not tuned to head direction
binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]
print(len(cellnames), cellnames)
# Remove nan items
whiches = np.isnan(headangle)
headangle = headangle[~whiches]
binnedspikes = binnedspikes[:,~whiches]

# Choose an interval of the total observed time
path = headangle[offset:offset+T]
binnedspikes = binnedspikes[:,offset:offset+T]
trackingtimes = trackingtimes[offset:offset+T]
print("How many times are there more than one spike:", sum((binnedspikes>1)*1))
if LIKELIHOOD_MODEL == "bernoulli":
    # Set spike counts to 0 or 1
    binnedspikes = (binnedspikes>0)*1
if (sum(isnan(path)) > 0):
    print("\nXXXXXXXXX\nXXXXXXXXX\nXXXXXXXXX\nThere are NAN values in path\nXXXXXXXXX\nXXXXXXXXX\nXXXXXXXXX")

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

# Find average observed spike count
print("Find average observed spike count")
number_of_X_bins = 50
bins = np.linspace(min(path)-0.000001, max(path) + 0.0000001, num=number_of_X_bins + 1)
evaluationpoints = 0.5*(bins[:(-1)]+bins[1:])
observed_spikes = zeros((N, number_of_X_bins))
# Observed spike average and estimated tuning curves
for i in range(N):
    for x in range(number_of_X_bins):
        timesinbin = (path>bins[x])*(path<bins[x+1])
        if(sum(timesinbin)>0):
            observed_spikes[i,x] = mean( y_spikes[i, timesinbin] )
## Plot average observed spike count
fig, ax = plt.subplots()
foo_mat = ax.matshow(y_spikes) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("y spikes")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-spikes-spike-count.png")

# Plotting spike activity 30.04.2020
#print("trackingtimes",trackingtimes[0:10])
# Plot binned spikes
plt.figure()
for i in range(len(cellnames)):
    plt.plot(trackingtimes-trackingtimes[0],binnedspikes[i,:]*(i+1), '|', color='black', markersize=1.)
    plt.ylabel("neuron")
    plt.xlabel("time")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-plot-spikes-y-spikes.png",format="png")
plt.show()
