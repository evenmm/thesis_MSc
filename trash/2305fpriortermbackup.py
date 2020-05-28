    ####################
    # f prior term #####
    ####################
    start = time.time()

    # Make (I - Kx B^-1 Kg)
    I_minus_KBK = np.identity(T) - np.matmul(K_xg, np.matmul(B_matrix_inverse, K_gx))   

    ## Split I_minus into its colums and put into tensor
    I_minus_column_tensor = I_minus_KBK.T.reshape((T, T, 1)) # Tensor with T depth containing single columns of length T 
    
    # Split I minus into rows
    I_minus_row_tensor = I_minus_KBK.reshape((T, 1, T)) # Tensor with T depth containing single rows of length N_ind 

    ## Make B_inv Kg
    B_inv_Kg = np.matmul(B_matrix_inverse, K_gx)

    ## Make tensor of copies of B inv Kg
    B_inv_Kg_tensor = np.repeat([B_inv_Kg],T,axis=0)

    ## dKx*B_inv_Kg tensor product
    dKx_B_inv_Kg = np.matmul(d_Kxg_tensor, B_inv_Kg_tensor)

    ## Make its transpose. All these are T by T by T
    Kx_B_inv_dKg = np.transpose(dKx_B_inv_Kg, axes=(0,2,1))

    startt = time.time()
    I_times_dK_B_K = np.matmul(I_minus_column_tensor, dKx_B_inv_Kg)
    stoptt = time.time() 
    print("I_times_dK_B_K", stoptt-startt)

    startt = time.time()
    K_B_dK_times_I = np.matmul(Kx_B_inv_dKg, I_minus_row_tensor)
    stoptt = time.time() 
    print("K_B_dK_times_I", stoptt-startt)

    squarebrackets = I_times_dK_B_K + K_B_dK_times_I

    ## Make copies of F_estimate
    F_tensor = np.repeat([F_estimate],T,axis=0)

    ## Do tensor product
    fM = np.matmul(F_tensor, squarebrackets)
    fMf = np.matmul(fM, np.transpose(F_tensor, axes=(0,2,1)))

    ## Trace for each matrix in the tensor
    fMfsum = np.trace(fMf, axis1=1, axis2=2)
    f_prior_gradient = sigma_n**-2 / 2 * fMfsum
    
    ## Elementwise
    #f_prior_gradient = np.zeros(T)
    #for t in range(T):
    #    d_Kxg_t = np.zeros((T,N_inducing_points))
    #    d_Kxg_t[t] = d_Kxg[t]
    #    # Make dKx B^-1 Kg and its transpose Kx B^-1 dKg 
    #    dKx_B_inv_Kg = np.matmul(d_Kxg_t, np.matmul(B_matrix_inverse, K_gx))
    #    Kx_B_inv_dKg = np.transpose(dKx_B_inv_Kg)
    #    # Add the two matrix products in the big brackets
    #    square_brackets = np.matmul(I_minus_KBK, dKx_B_inv_Kg) + np.matmul(Kx_B_inv_dKg, I_minus_KBK)
    #    # multiply by f on each side and sum over F using trace for computational speed
    #    fM = np.matmul(F_estimate, square_brackets)
    #    fMf = np.matmul(fM, F_estimate.T)
    #    fMfsum = np.trace(fMf)
    #    f_prior_gradient[t] = sigma_n**-2 / 2 * fMfsum
    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)

    ####################
    # f prior term, alternative take #####
    ####################
    start = time.time()

    Kgg_inv = np.linalg.inv(K_gg)

    # Make D = tempmatrix
    D = tempmatrix

    # make I - D
    IminusD = np.identity(T) - D

    # multply F_estimate by I - D
    FID = np.matmul(F_estimate, IminusD)

    # Transpose it
    IDF = np.transpose(FID)

    ###### C = dKKK + KKdK. Each must be handled separately
    # Here dKKK is handled
    dKKK = np.matmul(d_Kxg, np.matmul(Kgg_inv, K_gx))

    # Reshape dKKK to column tensor
    dKKK_column_tensor = dKKK.T.reshape((T,T,1))

    # Make copy_tensor of FID to multiply each row of dKKK with
    FID_copy_tensor = np.repeat([FID], T, axis=0)

    # Make row tensor of IDF to multiply FID_dKKK with in the end
    IDF_row_tensor = IDF.reshape((T,1,N))

    FID_dKKK = np.matmul(FID_copy_tensor, dKKK_column_tensor)

    FID_dKKK_IDF = np.matmul(FID_dKKK, IDF_row_tensor)

    fMfsum = np.trace(FID_dKKK_IDF, axis1=1, axis2=2)
    f_prior_gradient = sigma_n**-4 / 2 * fMfsum 

    ###### Handle KKdK
    # Here KKdK is handled
    KKdK = np.transpose(dKKK)

    # Reshape KKdK to row tensor
    KKdK_row_tensor = KKdK.reshape((T, 1, T))

    # Make copy tensor of IDF to multiply each row of KKdK with 
    IDF_copy_tensor = np.repeat([IDF], T, axis=0)

    # Make column tensor of FID to multiply KKdK_IDF with in the end
    FID_column_tensor = FID.T.reshape((T,N,1))

    KKdK_IDF = np.matmul(KKdK_row_tensor, IDF_copy_tensor)

    FID_KKdK_IDF = np.matmul(FID_column_tensor, KKdK_IDF)

    fMfsum = np.trace(FID_KKdK_IDF, axis1=1, axis2=2)
    f_prior_gradient += sigma_n**-4 / 2 * fMfsum 

    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)
"""
    ####################
    # f prior term, alternative take #####
    ####################
    start = time.time()

    Kgg_inv = np.linalg.inv(K_gg)

    # Make D = tempmatrix
    D = tempmatrix

    # make I - D
    IminusD = np.identity(T) - D

    # multply F_estimate by I - D
    FID = np.matmul(F_estimate, IminusD)

    # Transpose it
    IDF = np.transpose(FID)

    ###### C = dKKK + KKdK. Each must be handled separately
    # Here dKKK is handled
    dKKK = np.matmul(d_Kxg, np.matmul(Kgg_inv, K_gx))

    # Reshape dKKK to column tensor
    dKKK_column_tensor = dKKK.T.reshape((T,T,1))

    # Make copy_tensor of FID to multiply each row of dKKK with
    FID_copy_tensor = np.repeat([FID], T, axis=0)

    # Make row tensor of IDF to multiply FID_dKKK with in the end
    IDF_row_tensor = IDF.reshape((T,1,N))

    FID_dKKK = np.matmul(FID_copy_tensor, dKKK_column_tensor)

    FID_dKKK_IDF = np.matmul(FID_dKKK, IDF_row_tensor)

    fMfsum = np.trace(FID_dKKK_IDF, axis1=1, axis2=2)
    f_prior_gradient = sigma_n**-4 / 2 * fMfsum

    ###### Handle KKdK
    # Here KKdK is handled
    KKdK = np.transpose(dKKK)

    # Reshape KKdK to row tensor
    KKdK_row_tensor = KKdK.reshape((T, 1, T))

    # Make copy tensor of IDF to multiply each row of KKdK with 
    IDF_copy_tensor = np.repeat([IDF], T, axis=0)

    # Make column tensor of FID to multiply KKdK_IDF with in the end
    FID_column_tensor = FID.T.reshape((T,N,1))

    KKdK_IDF = np.matmul(KKdK_row_tensor, IDF_copy_tensor)

    FID_KKdK_IDF = np.matmul(FID_column_tensor, KKdK_IDF)

    fMfsum = np.trace(FID_KKdK_IDF, axis1=1, axis2=2)
    f_prior_gradient += sigma_n**-4 / 2 * fMfsum

    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)
"""

    ## Un-tensorized new hot take:
    # Elementwise in the sum, priority on things with dim T, AND things that don't need to be vectorized *first*.
    # Wrap things in from the sides to sandwich the tensor.
    f_Kx = np.matmul(F_estimate, K_xg)
    f_Kx_Binv = np.matmul(f_Kx, B_matrix_inverse)
    Binv_Kg_f = np.transpose(f_Kx_Binv)

    f_dKx = np.matmul(F_estimate, d_Kxg)
    dKg_f = np.transpose(f_dKx)

    Kg_dKx = np.matmul(K_gx, d_Kxg)
    dKg_Kx = np.transpose(Kg_dKx)

    ## f dKx Binv Kgx f
    fMf += np.matmul(f_dKx, Binv_Kg_f)

    ## f Kx Binv dKg Kx Binv Kg f
    fMf -= np.matmul(f_Kx_Binv, np.matmul(dKg_Kx, Binv_Kg_f))

    ## f Kx Binv Kg dKx Binv Kg f
    fMf -= np.matmul(f_Kx_Binv, np.matmul(Kg_dKx, Binv_Kg_f))

    ## f Kx Binv dKg f
    fMf += np.matmul(f_Kx_Binv, dKg_f)

    ## Trace for each matrix in the tensor
    fMfsum = np.trace(fMf, axis1=1, axis2=2)
    f_prior_gradient = sigma_n**-2 / 2 * fMfsum