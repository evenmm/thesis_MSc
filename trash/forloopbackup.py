# for loop variant

# Gradient of L 
def x_jacobian_no_la(X_estimate):
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
    smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
    # Kx_inducing_inverse = sigma_n**-2*np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
    tempmatrix = np.matmul(np.matmul(K_xg, smallinverse), K_gx)
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
    K_gx_tensor = K_gx.T.reshape((T, N_inducing_points, 1)) # Tensor with T depth containing single columns of length N_ind 
    d_Kxg_tensor = d_Kxg.reshape((T, 1, N_inducing_points)) # Tensor with T depth containing single rows of length N_ind 

    # Matrix multiply K_gx and d(K_xg)
    product_Kgx_dKxg = np.matmul(K_gx_tensor, d_Kxg_tensor) # 1000 by 30 by 30

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
    # f prior term #####
    ####################
    start = time.time()
    f_prior_gradient = np.zeros(T)

    # Make (I - Kx B^-1 Kg)
    I_minus_KBK = np.identity(T) - np.matmul(K_xg, np.matmul(B_matrix_inverse, K_gx))   

    # Elementwise
    for t in range(T):
        d_Kxg_t = np.zeros((T,N_inducing_points))
        d_Kxg_t[t] = d_Kxg[t]

        # Make dKx B^-1 Kg and its transpose Kx B^-1 dKg 
        dKx_B_inv_Kg = np.matmul(d_Kxg_t, np.matmul(B_matrix_inverse, K_gx))
        Kx_B_inv_dKg = np.transpose(dKx_B_inv_Kg)
        # Add the two matrix products in the big brackets
        square_brackets = np.matmul(I_minus_KBK, dKx_B_inv_Kg) + np.matmul(Kx_B_inv_dKg, I_minus_KBK)
        # multiply by f on each side and sum over F using trace for computational speed
        fM = np.matmul(F_estimate, square_brackets)
        fMf = np.matmul(fM, F_estimate.T)
        fMfsum = np.trace(fMf)
        f_prior_gradient[t] = sigma_n**-2 / 2 * fMfsum
    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)

    ####################
    # x prior term #####
    ####################
    start = time.time()
    x_prior_gradient = - np.dot(X_estimate.T, K_t_inverse)
    stop = time.time()
    if SPEEDCHECK:
        print("X prior term          :", stop-start)
    ####################

    #print("logdet_gradient\n", logdet_gradient)
    #print("f_prior_gradient\n",f_prior_gradient) # Is it sensible? Can plot these and compare with true X
    #print("x_prior_gradient\n", x_prior_gradient)

    x_gradient = logdet_gradient + f_prior_gradient + x_prior_gradient 
    return - x_gradient
