"""
    if iteration > 0:
        # Try variations of X_estimate 
        ensemble_points = 10
        linn = np.linspace(-0.3,0.3,ensemble_points)
        ensemble = [X_estimate + elem for elem in linn]
        Lvalues = [x_posterior_no_la(elem) for elem in ensemble]
        min_index = Lvalues.index(min(Lvalues))
    #    plt.figure()
    #    plt.title("Ensemble")
    #    plt.plot(X_estimate, label='Estimated')
    #    for i in range(ensemble_points):
    #        plt.plot(ensemble[i], label='Ensemble')
    #    plt.legend()
    #    plt.show()
        X_estimate = ensemble[min_index]
        plt.plot(X_estimate, label='Ensemble')

"""

"""
# Rescaling
sigma_n = 0.5
print("\n\nFind best scaled, offset X for sigma =",sigma_n)
initial_scale_offset = np.array([1,0])
def scaling(scale_offset):
    scaled_estimate = scale_offset[0] * X_estimate + scale_offset[1]
    return x_posterior_no_la(scaled_estimate)
scaling_optimization_result = optimize.minimize(scaling, initial_scale_offset, method = "L-BFGS-B", options = {'disp':True})
best_scale_offset = scaling_optimization_result.x
X_scaled = best_scale_offset[0] * X_estimate + best_scale_offset[1]
print("Best fit using sigma =", sigma_n, "is", best_scale_offset[0], "* X +", best_scale_offset[1])

plt.figure()
plt.title("Scaled&Shifted estimate")
plt.plot(path)
plt.plot(path, color="black", label='True X')
#plt.plot(X_estimate, label='Estimated')
plt.plot(X_scaled, label='Scaled&Shifted estimate')
plt.legend()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-scaled.png")
plt.show()
"""

# Shows how the tensor calculus makes sense
"""
#a = np.arange(2 * 3 * 4).reshape((4, 2, 3))
#print("a\n",a)
#print(np.matmul(a[0],a[1].T))
#b = np.arange(4 * 2 * 3).reshape((4, 3, 2))
#print("b\n",b)
#print("a dot b\n",np.matmul(a,b))

K = np.arange(T * 1 * N_inducing_points).reshape((N_inducing_points, T))
print("K\n",K)
K_tensor = K.T.reshape((T, N_inducing_points, 1))
print("K reshaped\n",K_tensor)

dK = np.arange(T * 1 * N_inducing_points).reshape((T, N_inducing_points))
print("dK\n",dK)
dK_tensor = dK.reshape((T, 1, N_inducing_points))
print("dK reshaped\n",dK_tensor)

prod = np.matmul(K_tensor, dK_tensor)
print("product\n",prod, "\n")


c = np.arange(T * 1 * N_inducing_points).reshape((T, N_inducing_points, 1))
print("c\n",c)
c_T = c.reshape(T,1,N_inducing_points)
print("c transpose\n", c_T)
print("product\n",np.matmul(c, c_T))
"""