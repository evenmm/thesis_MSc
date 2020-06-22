###########################################
# Find point estimates of hyperparameters #
###########################################
def hyperparam_loglikelihood(hyperparameters):
    sigma_x = hyperparameters[0]
    delta_x = hyperparameters[1]
    sigma_f_fit = hyperparameters[2]
    delta_f_fit = hyperparameters[3]
    #sigma_n = hyperparameters[4]
    # Create covariance matrices
    K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
    K_t = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_x, delta_x)
    K_t_inverse = np.linalg.inv(K_t)
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

    # logdet term ######
    ####################
    # My variant: 
    #logdet_term = - 0.5 * N * np.log(np.linalg.det(Kx_inducing))
    # Wu variant:
    logDetS1 = -np.log(np.linalg.det(smallinverse))-np.log(np.linalg.det(K_gg))+np.log(sigma_n)*(T-N_inducing_points)
    logdet_term = - 0.5 * N * logDetS1

    # x prior term #####
    ####################
    xTKt = np.dot(X_estimate.T, K_t_inverse) # Inversion trick for this too? No. If we don't do Fourier then we are limited by this.
    x_prior_term = - 0.5 * np.dot(xTKt, X_estimate)

    #print("f_prior_term",f_prior_term)
    #print("logdet_term",logdet_term)
    #print("x_prior_term",x_prior_term)
    posterior_loglikelihood = f_prior_term + logdet_term + x_prior_term
    #print("posterior_loglikelihood",posterior_loglikelihood)
    return - posterior_loglikelihood

if OPTIMIZE_HYPERPARAMETERS: 
    sigma_n = 1.5
    # length and sigma for X, length and sigma for f, noise param
    #initial_hyperparameters = [sigma_x, delta_x, sigma_f_fit, delta_f_fit, sigma_n] 
    initial_hyperparameters = [sigma_x, delta_x, sigma_f_fit, delta_f_fit] # sigma_n 
    bnds = ((0, None), (0, None), (0, None), (0, None))
    hyper_optim_result = optimize.minimize(hyperparam_loglikelihood, initial_hyperparameters, method = "L-BFGS-B", bounds=bnds, options = {'disp':True})
    optimized_hyperparameters = hyper_optim_result.x
    print("sigma_x:", optimized_hyperparameters[0])
    print("delta_x:", optimized_hyperparameters[1])
    print("sigma_f_fit:", optimized_hyperparameters[2])
    print("delta_f_fit:", optimized_hyperparameters[3])
    #print("sigma_n:", optimized_hyperparameters[4])
