from scipy import *
import scipy.io
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import cmocean
plt.rc('image', cmap='cmo.phase')

name = sys.argv[1] #'Mouse28-140313_stuff_BS0030_awakedata.mat'

sigma = 10 # window for smoothing
thresholdforneuronstokeep = 1000 # number of spikes to be considered useful

mat = scipy.io.loadmat(name)
headangle = ravel(array(mat['headangle']))
## plot head direction
plt.figure(figsize=(10,2))
plt.xlabel("Time")
plt.ylabel("x")
plt.tight_layout()
plt.plot(headangle, '.', color='black', markersize=1.)
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-headdirection-pca-path.png")
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-headdirection-pca-path.pdf",format="pdf")

cellspikes = array(mat['cellspikes'])
print("shape(cellspikes)", shape(cellspikes))
for i in range(len(cellspikes[0])):
  print(i, shape(cellspikes[0,i]))

cellnames = array(mat['cellnames'])
print("cellnames", shape(cellnames))
print(cellnames)

trackingtimes = ravel(array(mat['trackingtimes']))

## make matrix of spikes
startt = min(trackingtimes)
binsize = mean(trackingtimes[1:]-trackingtimes[:(-1)])
print(binsize)

nbins = len(trackingtimes)
print(nbins)

binnedspikes = zeros((len(cellnames), nbins))
celldata = zeros(shape(binnedspikes))
sgood = zeros(len(celldata[:,0]))<1
print(celldata[:,0])
print(len(celldata[:,0]))
print(sgood)

# Creating binnedspikes, a matrix with number of spikes in every timebin, 0 0 1 0 2 0 1. Created by seeing at what time every spike happens.
for i in range(len(cellnames)):
  spikes = ravel((cellspikes[0])[i]) #vector of times a neuron spikes
  for j in range(len(spikes)):
    # note 1ms binning means that number of ms from start is the correct index
    tt = int(floor(  (spikes[j] - startt)/float(binsize)  ))
    if(tt>nbins-1 or tt<0): # check if outside bounds of the awake time
      continue
    binnedspikes[i,tt] += 1 # add a spike to the thing

  ## check if actvitity is ok
  if(sum(binnedspikes[i,:])<thresholdforneuronstokeep):
      sgood[i] = False
      continue

  ## smooth and center the activity
  celldata[i,:] = scipy.ndimage.filters.gaussian_filter1d(binnedspikes[i,:], sigma)
  celldata[i,:] = (celldata[i,:]-mean(celldata[i,:]))/std(celldata[i,:])

celldata = celldata[sgood, :]
binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]

## From here on out it's just PCA and wrapping up goodbyes and kisses

## down-sample data so Isomap runs quickly (not necessary but important for us old people with little time left to live)
if(False):
  sss = mod(arange(len(celldata[0,:])),25)==0
  celldata = celldata[:,sss]
  headangle = headangle[sss]
  X = (Isomap(n_neighbors=50, n_components=2)).fit_transform(transpose(celldata))
else:
  X = (PCA(n_components=2, svd_solver='full')).fit_transform(transpose(celldata))

plt.figure(figsize=(12,10))

plt.clf()
whiches = np.isnan(headangle)
#plt.plot(X[whiches,0], X[whiches,1], '.', ms=5, color='black')

headangle = headangle[~whiches]
X = X[~whiches,:]

cs = headangle-min(headangle)
#cs /= max(cs)
sc = plt.scatter(X[:,0], X[:,1], s=3, c=cs, alpha=0.5, lw=0)
plt.colorbar(sc)
#plt.axis('off')
#plt.title('Using %d of %d neurons'%(sum(sgood), len(sgood)))
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-headdirection-pca.png")
plt.show()
