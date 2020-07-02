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

# Peyrache data with head direction neurons
if sys.argv[0] == "em-algorithm-peyrache-data.py":
    from parameter_file_peyrache import * 

# Robustness evaluation
elif sys.argv[0] == "cluster-parallel-robustness-evaluation.py":
    from parameter_file_robustness import *

# Example plotting
elif sys.argv[0] == "example_plotting.py":
    from parameter_file_exampleplotting import * 

######################
# Covariance kernels #
######################

def squared_exponential_covariance(xvector1, xvector2, sigma, delta):
    if COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = scipy.spatial.distance.cdist(xvector1, xvector2, 'sqeuclidean')
    if COVARIANCE_KERNEL_KX == "periodic":
        # This handles paths that stretches across anywhere as though the domain is truly periodic
        # First put every time point between 0 and 2pi
        xvector1 = xvector1 % (2*np.pi)
        xvector2 = xvector2 % (2*np.pi)
        # Then take care of periodicity
        distancesquared_1 = scipy.spatial.distance.cdist(xvector1, xvector2, 'sqeuclidean')
        distancesquared_2 = scipy.spatial.distance.cdist(xvector1+2*np.pi, xvector2, 'sqeuclidean')
        distancesquared_3 = scipy.spatial.distance.cdist(xvector1-2*np.pi, xvector2, 'sqeuclidean')
        min_1 = np.minimum(distancesquared_1, distancesquared_2)
        distancesquared = np.minimum(min_1, distancesquared_3)
    return sigma * exp(-distancesquared/(2*delta**2))

def exponential_covariance(tvector1, tvector2, sigma, delta):
    absolutedistance = scipy.spatial.distance.cdist(tvector1, tvector2, 'euclidean')
    return sigma * exp(-absolutedistance/delta)

########################
# Covariance matrices  #
########################
K_t = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_x, delta_x)
K_t_inverse = np.linalg.inv(K_t)

#########################
## Likelihood functions #
#########################

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize. Equation (4.17)++
def f_loglikelihood_bernoulli(f_i, sigma_n, y_i, K_xg_prev, K_gg): # Psi
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.log(1+np.exp(f_i))) # Corrected 16.03 from sum( np.multiply(y_i, (f_i - np.log(1+np.exp(f_i)))) + np.multiply((1-y_i), np.log(1- np.divide(np.exp(f_i), 1 + np.exp(f_i)))))
    priorterm_1 = -0.5*sigma_n**-2 * np.dot(f_i.T, f_i)
    fT_k = np.dot(f_i, K_xg_prev)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = 0.5*sigma_n**-2 * np.dot(np.dot(fT_k, smallinverse), fT_k.T)
    return - (likelihoodterm + priorterm_1 + priorterm_2)
def f_jacobian_bernoulli(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    yf_term = y_i - np.divide(np.exp(f_i), 1 + np.exp(f_i))
    priorterm_1 = -sigma_n**-2 * f_i
    kTf = np.dot(K_xg_prev.T, f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, kTf))
    f_derivative = yf_term + priorterm_1 + priorterm_2
    return - f_derivative
def f_hessian_bernoulli(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    e_tilde = np.divide(np.exp(f_i), (1 + np.exp(f_i))**2)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    f_hessian = - np.diag(e_tilde) - sigma_n**-2 * np.identity(T) + sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, K_xg_prev.T))
    return - f_hessian

# NEGATIVE Loglikelihood, gradient and Hessian. minimize to maximize.
def f_loglikelihood_poisson(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    likelihoodterm = sum( np.multiply(y_i, f_i) - np.exp(f_i)) 
    priorterm_1 = -0.5*sigma_n**-2 * np.dot(f_i.T, f_i)
    fT_k = np.dot(f_i, K_xg_prev)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = 0.5*sigma_n**-2 * np.dot(np.dot(fT_k, smallinverse), fT_k.T)
    return - (likelihoodterm + priorterm_1 + priorterm_2)

def f_jacobian_poisson(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    yf_term = y_i - np.exp(f_i)
    priorterm_1 = -sigma_n**-2 * f_i
    kTf = np.dot(K_xg_prev.T, f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    priorterm_2 = sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, kTf))
    f_derivative = yf_term + priorterm_1 + priorterm_2
    return - f_derivative
def f_hessian_poisson(f_i, sigma_n, y_i, K_xg_prev, K_gg):
    e_poiss = np.exp(f_i)
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_xg_prev.T, K_xg_prev))
    f_hessian = - np.diag(e_poiss) - sigma_n**-2*np.identity(T) + sigma_n**-2 * np.dot(K_xg_prev, np.dot(smallinverse, K_xg_prev.T))
    return - f_hessian

# L function
def x_posterior_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce): 
    start = time.time()
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T
    stop = time.time()
    if SPEEDCHECK:
        print("Speedcheck of L function:")
        print("Making Kxg            :", stop-start)

    start = time.time()
    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    stop = time.time()
    if SPEEDCHECK:
        print("Making small/tempmatrx:", stop-start)

    # yf_term ##########
    ####################
    #start = time.time()
    #if LIKELIHOOD_MODEL == "bernoulli": # equation 4.26
    #    yf_term = sum(np.multiply(y_spikes, F_estimate) - np.log(1 + np.exp(F_estimate)))
    #elif LIKELIHOOD_MODEL == "poisson": # equation 4.43
    #    yf_term = sum(np.multiply(y_spikes, F_estimate) - np.exp(F_estimate))
    #stop = time.time()
    #if SPEEDCHECK:
    #    print("yf term               :", stop-start)

    # f prior term #####
    ####################
    start = time.time()
    f_prior_term_1 = sigma_n**-2 * np.trace(np.matmul(F_estimate, F_estimate.T))
    fK = np.matmul(F_estimate, tempmatrix)
    fKf = np.matmul(fK, F_estimate.T)
    f_prior_term_2 = - sigma_n**-2 * np.trace(fKf)

    f_prior_term = - 0.5 * (f_prior_term_1 + f_prior_term_2)
    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)

    # logdet term ######
    ####################
    #logdet_term = - 0.5 * N * np.log(np.linalg.det(Kx_inducing))
    # smallinverse = np.linalg.inv(np.matmul(K_gx, K_xg) + K_gg*sigma_n**2)

    start = time.time()
    logDetS1 = np.log(np.linalg.det(np.matmul(K_gx, K_xg) + K_gg*sigma_n**2)) - np.log(np.linalg.det(K_gg)) + (T-N_inducing_points) * np.log(sigma_n**2)
    logdet_term = - 0.5 * N * logDetS1
    stop = time.time()
    if SPEEDCHECK:
        print("logdet term            :", stop-start)

    # x prior term #####
    ####################
    start = time.time()
    xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)
    stop = time.time()
    if SPEEDCHECK:
        print("X prior term          :", stop-start)
        print("logdet_term", logdet_term)
        print("f_prior_term", f_prior_term)
        print("x_prior_term", x_prior_term)
    posterior_loglikelihood = logdet_term + f_prior_term + x_prior_term #+ yf_term
    return - posterior_loglikelihood

# Gradient of L 
def x_jacobian_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    ####################
    # Initial matrices #
    ####################
    start = time.time()
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T
    stop = time.time()
    if SPEEDCHECK:
        print("\nSpeedcheck of x_jacobian function:")
        print("Making Kxg            :", stop-start)

    start = time.time()
    B_matrix = np.matmul(K_gx, K_xg) + (sigma_n**2) * K_gg
    B_matrix_inverse = np.linalg.inv(B_matrix)
    stop = time.time()
    if SPEEDCHECK:
        print("Making B and B inverse:", stop-start)

    start = time.time()
    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    #smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    stop = time.time()
    if SPEEDCHECK:
        print("Making small/tempmatrx:", stop-start)

    ####################
    # logdet term ######
    ####################
    start = time.time()

    ## Evaluate the derivative of K_xg. Row t of this matrix holds the nonzero row of the matrix d/dx_t K_xg
    d_Kxg = scipy.spatial.distance.cdist(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), lambda u, v: -(u-v)*np.exp(-(u-v)**2/(2*delta_f_fit**2)))
    d_Kxg = d_Kxg*sigma_f_fit*(delta_f_fit**-2)

    ## Reshape K_gx and K_xg to speed up matrix multiplication
    K_g_column_tensor = K_gx.T.reshape((T, N_inducing_points, 1)) # Tensor with T depth containing single columns of length N_ind 
    d_Kx_row_tensor = d_Kxg.reshape((T, 1, N_inducing_points)) # Tensor with T depth containing single rows of length N_ind 

    # Matrix multiply K_gx and d(K_xg)
    product_Kgx_dKxg = np.matmul(K_g_column_tensor, d_Kx_row_tensor) # 1000 by 30 by 30

    # Sum with transpose
    trans_sum_K_dK = product_Kgx_dKxg + np.transpose(product_Kgx_dKxg, axes=(0,2,1))

    # Create B^-1 copies for vectorial matrix multiplication
    B_inv_tensor = np.repeat([B_matrix_inverse],T,axis=0)
    
    # Then tensor multiply B^-1 with all the different trans_sum_K_dK
    big_tensor = np.matmul(B_inv_tensor, trans_sum_K_dK)
    
    # Take trace of each individually
    trace_array = np.trace(big_tensor, axis1=1, axis2=2)
    
    # Multiply by - N/2
    logdet_gradient = - N/2 * trace_array

    stop = time.time()
    if SPEEDCHECK:
        print("logdet term            :", stop-start)

    ####################
    # f prior term ##### (speeded up 10x)
    ####################
    start = time.time()
    fMf = np.zeros((T,N,N))

    ## New hot take:
    # Elementwise in the sum, priority on things with dim T, AND things that don't need to be vectorized *first*.
    # Wrap things in from the sides to sandwich the tensor.
    f_Kx = np.matmul(F_estimate, K_xg)
    f_Kx_Binv = np.matmul(f_Kx, B_matrix_inverse)
    #Binv_Kg_f = np.transpose(f_Kx_Binv)

    #d_Kg_column_tensor = np.transpose(d_Kx_row_tensor, axes=(0,2,1))

    # partial derivatives need tensorization
    # f_dKx = np.matmul(F_estimate, d_Kxg)
    f_column_tensor = F_estimate.T.reshape((T, N, 1))
    f_dKx_tensor = np.matmul(f_column_tensor, d_Kx_row_tensor) # (N x N_inducing) matrices  
    dKg_f_tensor = np.transpose(f_dKx_tensor, axes=(0,2,1))

    f_Kx_Binv_copy_tensor = np.repeat([f_Kx_Binv], T, axis=0)
    Binv_Kg_f_copy_tensor = np.transpose(f_Kx_Binv_copy_tensor, axes=(0,2,1)) #repeat([Binv_Kg_f], T, axis=0)

    ## A: f dKx Binv Kgx f
    fMf += np.matmul(f_dKx_tensor, Binv_Kg_f_copy_tensor)

    ## C: - f Kx Binv Kg dKx Binv Kg f
    Kg_dKx_tensor = np.matmul(K_g_column_tensor, d_Kx_row_tensor)
    f_Kx_Binv_Kg_dKx_tensor = np.matmul(f_Kx_Binv_copy_tensor, Kg_dKx_tensor)
    fMf -= np.matmul(f_Kx_Binv_Kg_dKx_tensor, Binv_Kg_f_copy_tensor)

    ## B: - f Kx Binv dKg Kx Binv Kg f
    dKg_Kx_tensor = np.transpose(Kg_dKx_tensor, axes=(0,2,1))
    f_Kx_Binv_dKg_Kx_tensor = np.matmul(f_Kx_Binv_copy_tensor, dKg_Kx_tensor)
    fMf -= np.matmul(f_Kx_Binv_dKg_Kx_tensor, Binv_Kg_f_copy_tensor)

    ## D: f Kx Binv dKg f
    fMf += np.matmul(f_Kx_Binv_copy_tensor, dKg_f_tensor)

    ## Trace for each matrix in the tensor
    fMfsum = np.trace(fMf, axis1=1, axis2=2)
    f_prior_gradient = sigma_n**(-2) / 2 * fMfsum

    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)

    ####################
    # x prior term #####
    ####################
    start = time.time()
    x_prior_gradient = (-1) * np.dot(X_estimate.T, K_t_inverse)
    stop = time.time()
    if SPEEDCHECK:
        print("X prior term          :", stop-start)
    ####################
    x_gradient = logdet_gradient + f_prior_gradient + x_prior_gradient 
    return - x_gradient

def just_fprior_term(X_estimate): 
    K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gx = K_xg.T
    #Kx_inducing = np.matmul(np.matmul(K_xg, K_gg_inverse), K_gx) + sigma_n**2
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)

    # f prior term #####
    ####################
    f_prior_term_1 = sigma_n**-2 * np.trace(np.matmul(F_estimate, F_estimate.T))
    fK = np.matmul(F_estimate, tempmatrix)
    fKf = np.matmul(fK, F_estimate.T)
    f_prior_term_2 = - sigma_n**-2 * np.trace(fKf)

    f_prior_term = - 0.5 * (f_prior_term_1 + f_prior_term_2)

    posterior_loglikelihood = f_prior_term #+ logdet_term #+ x_prior_term
    return - posterior_loglikelihood

###########################################################
##### Posterior inference of tuning curves on a grid ######
###########################################################

def posterior_f_inference(X_estimate, F_estimate, sigma_n, y_spikes, path, x_grid_for_plotting, bins_for_plotting, peak_f_offset, baseline_f_value):
    #X_estimate = path
    #print("Setting X_estimate = path for posterior F")

    if N_inducing_points == N_plotgridpoints:
        #################################################
        # Find posterior prediction of log tuning curve #
        #################################################

        # Inducing points (g efers to inducing points. Originally u did.)
        x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points)

        # K_xg = K_fu
        K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
        K_gx = K_xg.T

        # K_gg = K_uu and stands for inducing points
        K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
        # Adding tiny jitter term to diagonal of K_gg (not the same as sigma_n that we're adding to the diagonal of K_xgK_gg^-1K_gx later on)
        K_gg = K_gg_plain + jitter_term*np.identity(N_inducing_points)
        K_gg_inverse = np.linalg.inv(K_gg)

        # Plot K_gg inverse
        fig, ax = plt.subplots()
        kxmat = ax.matshow(K_gg_inverse, cmap=plt.cm.Blues)
        fig.colorbar(kxmat, ax=ax)
        plt.title("K_gg_inverse")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-posterior-f-infrence-K_gg_inverse.png")

        # Infer mean on the grid
        smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
        Q_xx_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
        Kxx_times_F = np.matmul(Q_xx_plus_sigma_inverse, F_estimate.T)
        #mu_posterior = np.matmul(Q_plotgrid_x, Kxx_times_F) # Here we have Kx crossover. Check what happens if swapped with Q = KKK
        mu_posterior = np.matmul(K_gx, Kxx_times_F)

        # Calculate standard deviations
        #sigma_posterior = K_plotgrid_plotgrid - np.matmul(Q_plotgrid_x, np.matmul(Q_xx_plus_sigma_inverse, Q_x_plotgrid))
        sigma_posterior = K_gg - np.matmul(K_gx, np.matmul(Q_xx_plus_sigma_inverse, K_xg))

    else:
        # If the number of plotgridpoints is different from inducing points, we do this: 
        ## A new grid is introduced here for plotting 

        #################################################
        # Find posterior prediction of log tuning curve #
        #################################################

        # Inducing points (g efers to inducing points. Originally u did.)
        x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points)

        # K_xg = K_fu
        K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
        K_gx = K_xg.T

        # K_gg = K_uu and stands for inducing points
        K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
        # Adding tiny jitter term to diagonal of K_gg (not the same as sigma_n that we're adding to the diagonal of K_xgK_gg^-1K_gx later on)
        K_gg = K_gg_plain + jitter_term*np.identity(N_inducing_points)
        K_gg_inverse = np.linalg.inv(K_gg)

        # Plot K_gg inverse
        fig, ax = plt.subplots()
        kxmat = ax.matshow(K_gg_inverse, cmap=plt.cm.Blues)
        fig.colorbar(kxmat, ax=ax)
        plt.title("K_gg_inverse")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-posterior-f-infrence-K_gg_inverse.png")

        # Connect x to plotgrid through inducing points
        K_g_plotgrid = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)
        K_plotgrid_g = K_g_plotgrid.T

        # Plot K_g_plotgrid
        fig, ax = plt.subplots()
        kx_cross_mat = ax.matshow(K_g_plotgrid, cmap=plt.cm.Blues)
        fig.colorbar(kx_cross_mat, ax=ax)
        plt.title("K_g_plotgrid")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-posterior-f-infrence-K_g_plotgrid.png")
        print("Making spatial covariance matrice: Kx grid")

        K_plotgrid_plotgrid = squared_exponential_covariance(x_grid_for_plotting.reshape((N_plotgridpoints,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)

        # Plot K_plotgrid_plotgrid
        fig, ax = plt.subplots()
        kxmat = ax.matshow(K_plotgrid_plotgrid, cmap=plt.cm.Blues)
        fig.colorbar(kxmat, ax=ax)
        plt.title("K_plotgrid_plotgrid")
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-posterior-f-infrence-K_plotgrid_plotgrid.png")

        Q_plotgrid_x = np.matmul(np.matmul(K_plotgrid_g, K_gg_inverse), K_gx)
        Q_x_plotgrid = Q_plotgrid_x.T

        # Infer mean on the grid
        smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
        Q_xx_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
        Kxx_times_F = np.matmul(Q_xx_plus_sigma_inverse, F_estimate.T)
        mu_posterior = np.matmul(Q_plotgrid_x, Kxx_times_F) # Here we have Kx crossover. Check what happens if swapped with Q = KKK

        # Calculate standard deviations
        sigma_posterior = K_plotgrid_plotgrid - np.matmul(Q_plotgrid_x, np.matmul(Q_xx_plus_sigma_inverse, Q_x_plotgrid))
        #########################################
        ### End of special treatment for different n.o. plotgridpoints ###
        ##################################################################

    # Plot posterior covariance matrix
    fig, ax = plt.subplots()
    sigma_posteriormat = ax.matshow(sigma_posterior, cmap=plt.cm.Blues)
    fig.colorbar(sigma_posteriormat, ax=ax)
    plt.title("Posterior covariance matrix")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-posterior-f-infrence-sigma_posterior.png")

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
    #if len(peak_lambda_array) > 1:
    #    print("NBNB! Take care which peak_lambda posterior F are found for!!!")
    #peak_lambda_global = peak_lambda_array[-1] 
    #peak_f_offset = np.log(peak_lambda_global) - baseline_f_value
    
    ## ONLY FOR SIMULATED DATA THAT HAS A BUMPFUNCTION
    #true_plot_f = np.zeros((N, N_plotgridpoints))
    #for i in range(N):
    #    for t in range(N_plotgridpoints):
    #        true_plot_f[i,t] = bumptuningfunction(x_grid_for_plotting[t], i, peak_f_offset)
    #true_expectation = np.exp(true_plot_f) #poisson

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
        plt.plot(x_grid_for_plotting, observed_mean_spikes_in_bins[i,:], color=plt.cm.viridis(0.1), label="Observed average")
        #plt.plot(x_grid_for_plotting, true_expectation[i,:], color=plt.cm.viridis(0.3), label="True expectation")
        #plt.plot(x_grid_for_plotting, h_estimate[i,:], color=plt.cm.viridis(0.5), label="Estimated expectation") 
        #plt.plot(x_grid_for_plotting, h_lower_confidence_limit[i,:], "--", color=plt.cm.viridis(0.5))
        #plt.plot(x_grid_for_plotting, h_upper_confidence_limit[i,:], "--", color=plt.cm.viridis(0.5))
        #plt.plot(x_grid_for_plotting, mu_posterior[i,:], color=plt.cm.viridis(0.5)) 
        #plt.title("Expected and average number of spikes, neuron "+str(i)) #spikes
        plt.title("Neuron "+str(i)+" with "+str(int(sum(y_spikes[i,:])))+" spikes")
        plt.yticks(range(0,1+math.ceil(max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:])))))
        plt.ylim(ymin=0., ymax=max(1, 1.05*math.ceil(max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:])))))
        #plt.yticks([0, max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:]))])
        plt.xlabel("x")
        plt.ylabel("Number of spikes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-posterior-f-infrence-tuning-"+str(i)+".png")

    # Plot observed tuning for all neurons together
    colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
    plt.figure()
    for i in range(N):
        plt.plot(x_grid_for_plotting, observed_mean_spikes_in_bins[i,:], color=colors[i])
    #    plt.plot(x_grid_for_plotting, h_estimate[neuron[i,j],:], color=plt.cm.viridis(0.5)) 
        plt.xlabel("x")
        plt.ylabel("Average number of spikes")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-posterior-f-infrence-tuning-collected.png")
    #plt.show()

