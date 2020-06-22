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
from multiprocessing import Pool
from sklearn.decomposition import PCA
from posterior_f_inference import * # inneholder function library og parameter file (enten Peyrache eller robustness)
print("Remember to use the rights parameter file")

##### Inferring actual HD in Peyrache data #####
## Parameters were tuned shamelessly to find a good fit...
## One smoothing window chosen for PCA must be chosen

# infer-peyrache-data.py
## History: 
## Formerly known as em-algorithm-peyrache-data.py
## Made before parallel-robustness-evaluation.py
## 16.06: Incorporate changes from parallel, apply to Peyrache data  

print("Likelihood model:",LIKELIHOOD_MODEL)
print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
print("Using gradient?", GRADIENT_FLAG)
print("Noise regulation:",NOISE_REGULARIZATION)
print("Initial sigma_n:", sigma_n)
print("Learning rate:", lr)
print("T:", T, "\n")
print("PCA smoothingwidth:", smoothingwindow_for_PCA)
if FLIP_AFTER_SOME_ITERATION:
    print("NBBBB!!! We're flipping the estimate after the second iteration in line 600.")
print("Offset:", offset)
print("Downsampling factor:", downsampling_factor)
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
#print("T_maximum", T_maximum)
if offset + T*downsampling_factor > T_maximum:
    sys.exit("Combination of offset, downsampling and T places the end of path outside T_maximum. Choose lower T, offset or downsampling factor.")

## 1) Remove headangles where the headangle value is NaN
# Spikes for Nan values are removed in step 2)
#print("How many NaN elements in path:", sum(np.isnan(path)))
whiches = np.isnan(path)
path = path[~whiches]

## 1.5) Make path continuous where it moves from 0 to 2pi
if not KEEP_PATH_BETWEEN_ZERO_AND_TWO_PI:
    for t in range(1,len(path)):
        if (path[t] - path[t-1]) < - np.pi:
            path[t:] += 2*np.pi
        if (path[t] - path[t-1]) > np.pi:
            path[t:] -= 2*np.pi

## 2) Since spikes are recorded as time points, we must make a matrix with counts 0,1,2,3,4
# Here we also remove spikes that happen at NaN headangles, and then we downsample the spike matrix by summing over bins
starttime = min(trackingtimes)
tracking_interval = mean(trackingtimes[1:]-trackingtimes[:(-1)])
#print("Observation frequency for path, and binsize for initial sampling:", tracking_interval)
binsize = tracking_interval
nbins = len(trackingtimes)
#print("Number of bins for entire interval:", nbins)
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

# Now remove spikes for NaN path values
binnedspikes = binnedspikes[:,~whiches]
# And downsample
binsize = downsampling_factor * tracking_interval
nbins = len(trackingtimes) // downsampling_factor
print("Bin size after downsampling: {:.2f}".format(binsize))
print("Number of bins for entire interval:", nbins)
print("Downsampling binned spikes...")
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

## plot head direction for the selected interval
if PLOTTING:
    if T > 100:
        plt.figure(figsize=(10,3))
    else:
        plt.figure()
    plt.plot(path, color="black", label='True X') #plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
    #plt.plot(trackingtimes, path, '.', color='black', markersize=1.) # trackingtimes as x optional
    #plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
    plt.xlabel("Time bin")
    plt.ylabel("x")
    plt.title("Head direction")
    #plt.yticks([0,3.14,6.28])
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-path.png")

## 5) Remove neurons that are not actually tuned to head direction
# On the entire range of time, these neurons are tuned to head direction:
active_and_strongly_tuned_from_70400_to_74400 = [68,63,53,45,39,38,37,36,35,34,33,31,29,27,26,25,23,22,21,20] #33 has few spikes
active_and_slightly_tuned_from_70400_to_74400 = [70,61,58,56,52,47,44,24,5,4]
barely_active_maybe_tuned = [69,64,62,60,28,18,17,3,2]
active_but_not_tuned_from_70400_to_74400 = [71,67,66,15,14,13,12,11,10,1]
#neuronsthataretunedtoheaddirection = [   17,18,   20,21,22,23,24,25,26,27,28,29,   31,32,34,35,36,37,38,39,68] # from my analysis and no spike cutoff
#                                     [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,            38, 47] ## froms tc-inference, after removing those with too few spikers
#neuronsthataretunedtoheaddirection = [17,18,19,20,21,22,23,24,25,26,27,29,31,34,35,36,38,39,68] # for presentation
#neuronsthataretunedtoheaddirection = [i for i in range(len(cellnames))] # all of them
#neuronsthataretunedtoheaddirection = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,47,68] # best of both worlds?
cutoff_spike_number = 50
sgood = np.zeros(len(cellnames))<1 
for i in range(len(cellnames)):
    #print(sum(binnedspikes[i,:]))
    if ((i not in active_and_strongly_tuned_from_70400_to_74400) or (sum(binnedspikes[i,:]) < cutoff_spike_number)):
        sgood[i] = False
binnedspikes = binnedspikes[sgood,:]
cellnames = cellnames[sgood]
print("len(cellnames)",len(cellnames))

# Plot binned spikes for selected neurons in the selected interval (Bernoulli style since they are binned)
bernoullispikes = (binnedspikes>0)*1
if PLOTTING:
    plt.figure(figsize=(5,4))
    for i in range(len(cellnames)):
        plt.plot(bernoullispikes[i,:]*(i+1), '|', color='black', markersize=2.)
        plt.ylabel("neuron")
        plt.xlabel("Time bin")
    plt.ylim(ymin=0.5)
    plt.yticks(range(1,len(cellnames)+1))
    #plt.yticks([9*i+1 for i in range(0,9)])
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-binnedspikes.png",format="png")
## 6) Change names to fit the rest of the code
#N = len(cellnames) #51 with cutoff at 1000 spikes
#print("N:",N)
if len(cellnames) != N:
    print("Wrong N. N must be set equal to" + str(N) + "in parameter_file_peyrache")
    sys.exit("N must be set equal to" + str(N) + "in parameter_file_peyrache")
y_spikes = binnedspikes
print("mean(y_spikes)",mean(y_spikes))
print("mean(y_spikes>0)",mean(y_spikes[y_spikes>0]))
# Spike distribution evaluation
spike_count = np.ndarray.flatten(binnedspikes)
#print("This is wrong: Portion of bins with more than one spike:", sum(spike_count>1)/T)
#print("This is wrong: Portion of nonzero bins with more than one:", sum(spike_count>1) / sum(spike_count>0)) 
# Remove zero entries:
#spike_count = spike_count[spike_count>0]
if PLOTTING:
    plt.figure()
    plt.hist(spike_count, bins=np.arange(0,int(max(spike_count))+1)-0.5, log=True, color=plt.cm.viridis(0.3))
    plt.ylabel("Number of bins")
    plt.xlabel("Spike count")
    plt.title("Spike histogram")
    plt.xticks(range(0,int(max(spike_count)),1))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-spike-histogram-log.png")

# Plot y spikes
if PLOTTING:
    fig, ax = plt.subplots(figsize=(8,1))
    foo_mat = ax.matshow(y_spikes) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("y spikes")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-y-spikes.png")

# Inducing points based on a predetermined range
x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points) #np.linspace(min(path), max(path), N_inducing_points)
print("Min and max of path:", min(path), max(path))

K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
if PLOTTING:
    fig, ax = plt.subplots()
    foo_mat = ax.matshow(K_gg_plain, cmap=plt.cm.Blues)
    fig.colorbar(foo_mat, ax=ax)
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-kgg.png")
K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)

# Grid for plotting
bins_for_plotting = np.linspace(0, 2*np.pi, num=N_plotgridpoints + 1)
x_grid_for_plotting = 0.5*(bins_for_plotting[:(-1)]+bins_for_plotting[1:])

######################
# Initialize X and F #
######################
# PCA initialization: 
celldata = zeros(shape(y_spikes))
for i in range(N):
    celldata[i,:] = scipy.ndimage.filters.gaussian_filter1d(y_spikes[i,:], smoothingwindow_for_PCA) # smooth
    #celldata[i,:] = (celldata[i,:]-mean(celldata[i,:]))/std(celldata[i,:])                 # standardization requires at least one spike
X_pca_result_2comp = PCA(n_components=2, svd_solver='full').fit_transform(transpose(celldata))
X_pca_result_1comp = PCA(n_components=1, svd_solver='full').fit_transform(transpose(celldata))
X_pca_result_1comp 
pca_radii = np.sqrt(X_pca_result_2comp[:,0]**2 + X_pca_result_2comp[:,1]**2)
pca_angles = np.arccos(X_pca_result_2comp[:,0]/pca_radii)
X_pca_initial = np.zeros(T)
for i in range(T):
    if PCA_TYPE == "angle":
        X_pca_initial[i] = pca_angles[i]
    elif PCA_TYPE == "1d":
        X_pca_initial[i] = X_pca_result_1comp[i]
# Scale PCA initialization to fit domain:
if KEEP_PATH_BETWEEN_ZERO_AND_TWO_PI:
    X_pca_initial -= min(X_pca_initial)
    X_pca_initial /= max(X_pca_initial)
    X_pca_initial *= 2*np.pi
    X_pca_initial += 0
# Flip PCA initialization correctly by comparing to true path
X_pca_initial_flipped = 2*mean(X_pca_initial) - X_pca_initial
X_pca_initial_rmse = np.sqrt(sum((X_pca_initial-path)**2) / T)
X_pca_initial_flipped_rmse = np.sqrt(sum((X_pca_initial_flipped-path)**2) / T)
if X_pca_initial_flipped_rmse < X_pca_initial_rmse:
    X_pca_initial = X_pca_initial_flipped
# Plot PCA initialization
if T > 100:
    plt.figure(figsize=(10,3))
else:
    plt.figure()
plt.xlabel("Time bin")
plt.ylabel("x")
plt.title("PCA initial of X")
plt.plot(path, color="black", label='True X')
plt.plot(X_pca_initial, label="Initial")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-PCA-initial.png")


# Initialize X
np.random.seed(0)
if X_initialization == "true":
    X_initial = path
if X_initialization == "ones":
    X_initial = np.ones(T)
if X_initialization == "pca":
    X_initial = X_pca_initial
if X_initialization == "randomrandom":
    X_initial = (max_inducing_point - min_inducing_point)*np.random.random(T)
if X_initialization == "randomprior":
    X_initial = (max_inducing_point - min_inducing_point)*np.random.multivariate_normal(np.zeros(T), K_t)
if X_initialization == "linspace":
    X_initial = np.linspace(min_inducing_point, max_inducing_point, T) 
if X_initialization == "supreme":
    X_initial = np.load("X_estimate_supreme.npy")
if X_initialization == "flatrandom":
    X_initial = 1.5*np.ones(T) + 0.2*np.random.random(T)

X_estimate = np.copy(X_initial)

if PLOTTING:
    if T > 100:
        plt.figure(figsize=(10,3))
    else:
        plt.figure()
    plt.title("Initial X")
    plt.xlabel("Time bin")
    plt.ylabel("x")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    #plt.legend(loc="upper right")
    #plt.ylim((0, 2*np.pi))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-X-initial.png")

# Initialize F
F_initial = np.sqrt(y_spikes) - np.amax(np.sqrt(y_spikes))/2 #np.log(y_spikes + 0.0008)
F_estimate = np.copy(F_initial)
if X_initialization == "supreme":
    print("Initializing F supremely too")
    F_initial = np.load("F_estimate_supreme.npy")
F_estimate = np.copy(F_initial)

if GIVEN_TRUE_F:
    # Initialize F at the values given path:
    print("Setting f hat to the estimates given the true path")
    temp_X_estimate = np.copy(X_estimate)
    X_estimate = path
    K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    if LIKELIHOOD_MODEL == "bernoulli":
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_bernoulli, F_estimate[i], jac=f_jacobian_bernoulli, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
            F_estimate[i] = optimization_result.x
    elif LIKELIHOOD_MODEL == "poisson":
        for i in range(N):
            y_i = y_spikes[i]
            optimization_result = optimize.minimize(f_loglikelihood_poisson, F_estimate[i], jac=f_jacobian_poisson, method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
            F_estimate[i] = optimization_result.x 
    true_f = np.copy(F_estimate)
    ## Plot F estimate
    if PLOTTING:
        fig, ax = plt.subplots(figsize=(10,1))
        foo_mat = ax.matshow(F_estimate) #cmap=plt.cm.Blues
        fig.colorbar(foo_mat, ax=ax)
        plt.title("F given path")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-F-optimal.png")
    X_estimate = temp_X_estimate

## Plot initial f
if PLOTTING:
    fig, ax = plt.subplots(figsize=(8,1))
    foo_mat = ax.matshow(F_initial) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("Initial f")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-initial-f.png")

if PLOTTING:
    if T > 100:
        plt.figure(figsize=(10,3))
    else:
        plt.figure()
    plt.title("X estimate")
    plt.xlabel("Time bin")
    plt.ylabel("x")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    #plt.legend(loc="upper right")
    #plt.ylim((0, 2*np.pi))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-X-estimate.png")

collected_estimates = np.zeros((N_iterations, T))
prev_X_estimate = np.Inf
### EM algorithm: Find f given X, then X given f.
for iteration in range(N_iterations):
    if iteration > 0:
        sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
        if LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE:
            x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) # Change position of grid to position of estimate
    K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
    K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    # Find F estimate only if we're not at the first iteration
    if iteration > 0:
        if LIKELIHOOD_MODEL == "bernoulli":
            for i in range(N):
                y_i = y_spikes[i]
                optimization_result = optimize.minimize(fun=f_loglikelihood_bernoulli, x0=F_estimate[i], jac=f_jacobian_bernoulli, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
                F_estimate[i] = optimization_result.x
        elif LIKELIHOOD_MODEL == "poisson":
            for i in range(N):
                y_i = y_spikes[i]
                optimization_result = optimize.minimize(fun=f_loglikelihood_poisson, x0=F_estimate[i], jac=f_jacobian_poisson, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
                F_estimate[i] = optimization_result.x 
    # Find next X estimate, that can be outside (0,2pi)
    if GIVEN_TRUE_F: 
        print("NB! NB! We're setting the f value to the optimal F given the path.")
        F_estimate = np.copy(true_f)
    if NOISE_REGULARIZATION:
        X_estimate += 2*np.random.multivariate_normal(np.zeros(T), K_t) - 1
    if SMOOTHING_REGULARIZATION and iteration < (N_iterations-1) :
        X_estimate = scipy.ndimage.filters.gaussian_filter1d(X_estimate, 4)
    if GRADIENT_FLAG: 
        optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
    else:
        optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
    X_estimate = optimization_result.x
    if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
        # Flipping estimate after iteration 1 has been plotted
        X_estimate = 2*mean(X_estimate) - X_estimate
    if USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION:
        X_estimate -= min(X_estimate) #set offset of min to 0
        X_estimate /= max(X_estimate) #scale length to 1
        X_estimate *= (max(path)-min(path)) #scale length to length of path
        X_estimate += min(path) #set offset to offset of path
    if PLOTTING:
        plt.plot(X_estimate, label='Estimate')
        #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-X-estimate.png")
    if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
        break
    prev_X_estimate = X_estimate
    #np.save("X_estimate", X_estimate)
if USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE:
    X_estimate -= min(X_estimate) #set offset of min to 0
    X_estimate /= max(X_estimate) #scale length to 1
    X_estimate *= (max(path)-min(path)) #scale length to length of path
    X_estimate += min(path) #set offset to offset of path
# Flipped 
X_flipped = - X_estimate + 2*mean(X_estimate)
# Rootmeansquarederror for X
X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
X_flipped_rmse = np.sqrt(sum((X_flipped-path)**2) / T)
##### Check if flipped and maybe iterate again with flipped estimate
if X_flipped_rmse < X_rmse:
    #print("RMSE for X:", X_rmse)
    #print("RMSE for X flipped:", X_flipped_rmse)
    #print("Re-iterating because of flip")
    x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points) #np.linspace(min(path), max(path), N_inducing_points)
    K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    X_initial_2 = np.copy(X_flipped)
    X_estimate = np.copy(X_flipped)
    F_estimate = np.copy(F_initial)
    if GIVEN_TRUE_F:
        F_estimate = true_f
    if PLOTTING:
        if T > 100:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        #plt.title("After flipping") # as we go
        plt.xlabel("Time bin")
        plt.ylabel("x")
        plt.plot(path, color="black", label='True X')
        plt.plot(X_initial_2, label='Initial')
        #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-X-flipped.png")
    #############################
    # EM after flipped          #
    #############################
    prev_X_estimate = np.Inf
    sigma_n = np.copy(global_initial_sigma_n)
    for iteration in range(N_iterations):
        if iteration > 0:
            sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
            if LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE:
                x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) # Change position of grid to position of estimate
        K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
        K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
        # Find F estimate only if we're not at the first iteration
        if iteration > 0:
            if LIKELIHOOD_MODEL == "bernoulli":
                for i in range(N):
                    y_i = y_spikes[i]
                    optimization_result = optimize.minimize(fun=f_loglikelihood_bernoulli, x0=F_estimate[i], jac=f_jacobian_bernoulli, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
                    F_estimate[i] = optimization_result.x
            elif LIKELIHOOD_MODEL == "poisson":
                for i in range(N):
                    y_i = y_spikes[i]
                    optimization_result = optimize.minimize(fun=f_loglikelihood_poisson, x0=F_estimate[i], jac=f_jacobian_poisson, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
                    F_estimate[i] = optimization_result.x 
        # Find next X estimate, that can be outside (0,2pi)
        if GIVEN_TRUE_F: 
            print("NB! NB! We're setting the f value to the optimal F given the path.")
            F_estimate = np.copy(true_f)
        if NOISE_REGULARIZATION:
            X_estimate += 2*np.random.multivariate_normal(np.zeros(T), K_t) - 1
        if SMOOTHING_REGULARIZATION and iteration < (N_iterations-1) :
            X_estimate = scipy.ndimage.filters.gaussian_filter1d(X_estimate, 4)
        if GRADIENT_FLAG: 
            optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
        else:
            optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
        X_estimate = optimization_result.x
        if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
            # Flipping estimate after iteration 1 has been plotted
            X_estimate = 2*mean(X_estimate) - X_estimate
        if USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION:
            X_estimate -= min(X_estimate) #set offset of min to 0
            X_estimate /= max(X_estimate) #scale length to 1
            X_estimate *= (max(path)-min(path)) #scale length to length of path
            X_estimate += min(path) #set offset to offset of path
        if PLOTTING:
            plt.plot(X_estimate, label='Estimate (after flip)')
            #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-X-flipped.png")
        if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
            #print("Seed", seeds[seedindex], "Iterations after flip:", iteration+1, "Change in X smaller than TOL")
            break
        #if iteration == N_iterations-1:
        #    print("Seed", seeds[seedindex], "Iterations after flip:", iteration+1, "N_iterations reached")
        prev_X_estimate = X_estimate
    if USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE:
        X_estimate -= min(X_estimate) #set offset of min to 0
        X_estimate /= max(X_estimate) #scale length to 1
        X_estimate *= (max(path)-min(path)) #scale length to length of path
        X_estimate += min(path) #set offset to offset of path
    # Rootmeansquarederror for X
    X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
######################
#### Handle rotation #
######################
SStot = sum((path - mean(path))**2)
SSdev = sum((X_estimate-path)**2)
Rsquared = 1 - SSdev / SStot
print("R squared value of X estimate:", Rsquared)

if PLOTTING:
    if T > 100:
        plt.figure(figsize=(10,3))
    else:
        plt.figure()
    plt.title("Final estimate") # as we go
    plt.xlabel("Time bin")
    plt.ylabel("x")
    plt.plot(path, color="black", label='True X')
    plt.plot(X_initial, label='Initial')
    plt.plot(X_estimate, label='Estimate')
    plt.legend(loc="upper right")
    #plt.ylim((min_neural_tuning_X, max_neural_tuning_X))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-peyrache-X-final.png")

#################################################
# Find posterior prediction of log tuning curve #
#################################################
if INFER_F_POSTERIORS:
    posterior_f_inference(F_estimate, 1, y_spikes, path , X_estimate)
