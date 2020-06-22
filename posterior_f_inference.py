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
from function_library import * # loglikelihoods, gradients, covariance functions, tuning curve definitions

###########################################################
##### Posterior inference of tuning curves on a grid ######
###########################################################



def posterior_f_inference(F_estimate, sigma_n, y_spikes, path, X_estimate):
    #X_estimate = path
    #print("Setting X_estimate = path for posterior F")

    ## X_estimate is used to make Kxg.
    ## A new grid is introduced here for plotting (not really necessary but it works)
    #################################################
    # Find posterior prediction of log tuning curve #
    #################################################

    # Inducing points (g efers to inducing points. Originally u did.)
    x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points)

    # Grid for plotting
    bins_for_plotting = np.linspace(lower_domain_limit, upper_domain_limit, num=N_plotgridpoints + 1)
    x_grid_for_plotting = 0.5*(bins_for_plotting[:(-1)]+bins_for_plotting[1:])

    # K_xg = K_fu
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T

    # K_gg = K_uu and stands for inducing points
    K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    # Adding tiny jitter term to diagonal of K_gg (not the same as sigma_n that we're adding to the diagonal of K_xgK_gg^-1K_gx later on)
    K_gg = K_gg_plain + jitter_term*np.identity(N_inducing_points)
    K_gg_inverse = np.linalg.inv(K_gg)

    # Connect x to plotgrid through inducing points
    K_g_plotgrid = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)
    K_plotgrid_g = K_g_plotgrid.T

    # Plot K_g_plotgrid
    fig, ax = plt.subplots()
    kx_cross_mat = ax.matshow(K_g_plotgrid, cmap=plt.cm.Blues)
    fig.colorbar(kx_cross_mat, ax=ax)
    plt.title("K_g_plotgrid")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-K_g_plotgrid.png")
    print("Making spatial covariance matrice: Kx grid")

    K_plotgrid_plotgrid = squared_exponential_covariance(x_grid_for_plotting.reshape((N_plotgridpoints,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)

    # Plot K_plotgrid_plotgrid
    fig, ax = plt.subplots()
    kxmat = ax.matshow(K_plotgrid_plotgrid, cmap=plt.cm.Blues)
    fig.colorbar(kxmat, ax=ax)
    plt.title("K_plotgrid_plotgrid")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-K_plotgrid_plotgrid.png")

    Q_plotgrid_x = np.matmul(np.matmul(K_plotgrid_g, K_gg_inverse), K_gx)
    Q_x_plotgrid = Q_plotgrid_x.T

    # Infer mean on the grid
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    Q_xx_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    Kxx_times_F = np.matmul(Q_xx_plus_sigma_inverse, F_estimate.T)
    mu_posterior = np.matmul(Q_plotgrid_x, Kxx_times_F) # Here we have Kx crossover. Check what happens if swapped with Q = KKK

    # Calculate standard deviations
    sigma_posterior = K_plotgrid_plotgrid - np.matmul(Q_plotgrid_x, np.matmul(Q_xx_plus_sigma_inverse, Q_x_plotgrid))

    # Plot posterior covariance matrix
    fig, ax = plt.subplots()
    sigma_posteriormat = ax.matshow(sigma_posterior, cmap=plt.cm.Blues)
    fig.colorbar(sigma_posteriormat, ax=ax)
    plt.title("Posterior covariance matrix")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-sigma_posterior.png")

    ###############################################
    # Plot tuning curve with confidence intervals #
    ###############################################
    standard_deviation = [np.sqrt(np.diag(sigma_posterior))]
    print("posterior marginal standard deviation:\n",standard_deviation[0])
    standard_deviation = np.repeat(standard_deviation, N, axis=0)
    upper_confidence_limit = mu_posterior + 1.96*standard_deviation.T
    lower_confidence_limit = mu_posterior - 1.96*standard_deviation.T

    if LIKELIHOOD_MODEL == "bernoulli":
        h_estimate = np.divide( np.exp(mu_posterior), (1 + np.exp(mu_posterior)))
        h_upper_confidence_limit = np.exp(upper_confidence_limit) / (1 + np.exp(upper_confidence_limit))
        h_lower_confidence_limit = np.exp(lower_confidence_limit) / (1 + np.exp(lower_confidence_limit))
    if LIKELIHOOD_MODEL == "poisson":
        h_estimate = np.exp(mu_posterior)
        h_upper_confidence_limit = np.exp(upper_confidence_limit)
        h_lower_confidence_limit = np.exp(lower_confidence_limit)

    mu_posterior = mu_posterior.T
    h_estimate = h_estimate.T
    h_upper_confidence_limit = h_upper_confidence_limit.T
    h_lower_confidence_limit = h_lower_confidence_limit.T

    ## Find true rate on plotgrid
    if len(peak_lambda_array) > 1:
        print("NBNB! Take care which peak_lambda posterior F are found for!!!")
    peak_lambda_global = peak_lambda_array[-1] 
    peak_f_offset = np.log(peak_lambda_global) - baseline_f_value
    true_plot_f = np.zeros((N, N_plotgridpoints))
    for i in range(N):
        for t in range(N_plotgridpoints):
            true_plot_f[i,t] = bumptuningfunction(x_grid_for_plotting[t], i, peak_f_offset)

    true_expectation = np.exp(true_plot_f) #poisson

    ## Find observed firing rate
    observed_mean_spikes_in_bins = zeros((N, N_plotgridpoints))
    for i in range(N):
        for x in range(N_plotgridpoints):
            timesinbin = (path>bins_for_plotting[x])*(path<bins_for_plotting[x+1])
            if(sum(timesinbin)>0):
                observed_mean_spikes_in_bins[i,x] = mean( y_spikes[i, timesinbin] )
            elif i==0:
                print("No observations of X between",bins_for_plotting[x],"and",bins_for_plotting[x+1],".")
    for i in range(N):
        plt.figure()
        #plt.plot(x_grid_for_plotting, observed_mean_spikes_in_bins[i,:], color=plt.cm.viridis(0.1), label="Observed average")
        plt.plot(x_grid_for_plotting, true_expectation[i,:], color=plt.cm.viridis(0.3), label="True expectation")
        plt.plot(x_grid_for_plotting, h_estimate[i,:], color=plt.cm.viridis(0.5), label="Estimated expectation") 
        plt.plot(x_grid_for_plotting, h_lower_confidence_limit[i,:], "--", color=plt.cm.viridis(0.5))
        plt.plot(x_grid_for_plotting, h_upper_confidence_limit[i,:], "--", color=plt.cm.viridis(0.5))
    #    plt.plot(x_grid_for_plotting, mu_posterior[i,:], color=plt.cm.viridis(0.5)) 
        plt.title("Expected and average number of spikes, neuron "+str(i)) #spikes
    #    plt.title("Neuron "+str(i)+" with "+str(sum(y_spikes[i,:]))+" spikes")
        plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:])))
        plt.yticks(range(0,math.floor(max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:])))))
        plt.xlabel("x")
        plt.ylabel("Number of spikes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-tuning-"+str(i)+".png")

    # Plot observed tuning for all neurons together
    colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
    plt.figure()
    for i in range(N):
        plt.plot(x_grid_for_plotting, observed_mean_spikes_in_bins[i,:], color=colors[i])
    #    plt.plot(x_grid_for_plotting, h_estimate[neuron[i,j],:], color=plt.cm.viridis(0.5)) 
        plt.xlabel("x")
        plt.ylabel("Average number of spikes")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-tuning-collected.png")
    #plt.show()

