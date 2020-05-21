T = 4
N_inducing_points = 2

K = np.arange(T * 1 * N_inducing_points).reshape((N_inducing_points, T))
K_tensor = K.T.reshape((T, N_inducing_points, 1))
print("K tensor\n",K_tensor)
#print("np.sum(K_tensor, axis=1)\n", np.sum(K_tensor, axis=2))

dK = np.arange(T * 1 * N_inducing_points).reshape((T, N_inducing_points))
print("dK\n",dK)
dK_tensor = dK.reshape((T, 1, N_inducing_points))
print("dK tensor\n",dK_tensor)

prod = np.matmul(K_tensor, dK_tensor)
print("product\n",prod, "\n")
print("transpose\n",np.transpose(prod, axes=(0,2,1)), "\n")
trans_sum_K_dK = prod + np.transpose(prod, axes=(0,2,1))
print("sum\n",trans_sum_K_dK)

e = [trans_sum_K_dK[0]]
print("e\n",e)
print(np.repeat(e,T,axis=0))

arrr = np.matmul(e, trans_sum_K_dK)
print("arrr\n",arrr)
print("trace", np.trace(arrr, axis1=1, axis2=2))

print(np.array([[0,1],[2,3]]))
print(np.transpose([[0,1],[2,3]]))



"""
#a = np.arange(2 * 3 * 4).reshape((4, 2, 3))
#print("a\n",a)
#print(np.matmul(a[0],a[1].T))
#b = np.arange(4 * 2 * 3).reshape((4, 3, 2))
#print("b\n",b)
#print("a dot b\n",np.matmul(a,b))




c = np.arange(T * 1 * N_inducing_points).reshape((T, N_inducing_points, 1))
print("c\n",c)
c_T = c.reshape(T,1,N_inducing_points)
print("c transpose\n", c_T)
print("product\n",np.matmul(c, c_T))
"""
####################
"""
xvector = np.arange(T).reshape((T,1))
xgrid = np.arange(N_inducing_points).reshape((N_inducing_points,1))
start = time.time()
distancesquared_1 = scipy.spatial.distance.cdist(xgrid, xvector, 'sqeuclidean')
stop = time.time()
print("sqeuclidian            :", stop-start)
print(distancesquared_1)
sigma = 0.5
delta = 0.3

#differentiated_distance_kernel = lambda u, v: -(u-v)*sigma*(delta**-2)*np.exp(-(u-v)**2/(2*delta**2))
start = time.time()
#distancesquared_1 = scipy.spatial.distance.cdist(xgrid, xvector, lambda u, v: ((u-v)**2))
distancesquared_1 = scipy.spatial.distance.cdist(xgrid, xvector, lambda u, v: -(u-v)*np.exp(-(u-v)**2/(2*delta**2)))
distancesquared_1 *= sigma*(delta**-2) # saves us some time 
stop = time.time()
print("homemade            :", stop-start)
print(distancesquared_1)
"""


    # f prior term before tensor
    ####################
    # f prior term #####
    ####################
    start = time.time()

    # Make (I - Kx B^-1 Kg)
    I_minus_KBK = np.identity(T) - np.matmul(K_xg, np.matmul(B_matrix_inverse, K_gx))   

    ##########

    # Make dKx B^-1 Kg and its transpose Kx B^-1 dKg 
    dKx_B_inv_Kg = np.matmul(d_Kxg, np.matmul(B_matrix_inverse, K_gx))
    Kx_B_inv_dKg = np.transpose(dKx_B_inv_Kg)

    # Add the two matrix products in the big brackets
    square_brackets = np.matmul(I_minus_KBK, dKx_B_inv_Kg) + np.matmul(Kx_B_inv_dKg, I_minus_KBK)

    # multiply by f on each side and sum over F using trace for computational speed
    fM = np.matmul(F_estimate, square_brackets)
    fMf = np.matmul(fM, F_estimate.T)
    fMfsum = np.trace(fMf)
    f_prior_gradient = sigma_n**-2 / 2 * fMfsum

    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)



## After tensor attempt ####
    ####################
    # f prior term #####
    ####################
    start = time.time()

    # Make (I - Kx B^-1 Kg)
    I_minus_KBK = np.identity(T) - np.matmul(K_xg, np.matmul(B_matrix_inverse, K_gx))
    I_tensor = np.repeat([I_minus_KBK],T,axis=0)
    print(shape(I_tensor))

    # Make dKx B^-1 Kg and its transpose Kx B^-1 dKg 
    dKx_B_inv_Kg = np.matmul(d_Kxg_tensor, np.matmul(B_inv_tensor, K_gx_tensor))
    Kx_B_inv_dKg = np.transpose(dKx_B_inv_Kg, axes=(0,2,1))
    print(shape(dKx_B_inv_Kg))
    print(shape(Kx_B_inv_dKg))
    plt.show()
    ###
    K_gx_tensor = K_gx.T.reshape((T, N_inducing_points, 1)) # Tensor with T depth containing single columns of length N_ind 
    d_Kxg_tensor = d_Kxg.reshape((T, 1, N_inducing_points)) # Tensor with T depth containing single rows of length N_ind 

    # Add the two matrix products in the big brackets
    square_brackets = np.matmul(I_minus_KBK, dKx_B_inv_Kg) + np.matmul(Kx_B_inv_dKg, I_minus_KBK)

    # multiply by f on each side and sum over F using trace for computational speed
    fM = np.matmul(F_estimate, square_brackets)
    fMf = np.matmul(fM, F_estimate.T)
    fMfsum = np.trace(fMf)
    f_prior_gradient = sigma_n**-2 / 2 * fMfsum

    stop = time.time()
    if SPEEDCHECK:
        print("f prior term          :", stop-start)
