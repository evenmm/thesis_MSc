
F_estimate = F_array[0]
sigma_n = 1
y_spikes = Y_array[0]
path = path_array[0] 
X_estimate = X_array[0]

#X_estimate = path
print("X_estimate = path")

## Stance taken: X_estimate must be used for Kxg.
## There is a new grid being introduced here for plotting. 
#################################################
# Find posterior prediction of log tuning curve #
#################################################

#def exponential_covariance(t1,t2, sigma, delta):
#    distance = abs(t1-t2)
#    return sigma * exp(-distance/delta)
#
#def gaussian_periodic_covariance(x1,x2, sigma, delta):
#    distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
#    return sigma * exp(-distancesquared/(2*delta))
#
#def gaussian_NONPERIODIC_covariance(x1,x2, sigma, delta):
#    distancesquared = (x1-x2)**2
#    return sigma * exp(-distancesquared/(2*delta))

# Inducing points (g above and below refers to inducing points. Originally u did.)
x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points)

# K_xg = K_fu
K_xg = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
K_gx = K_xg.T
#K_fu = np.zeros((T,N_inducing_points))
#for x1 in range(T):
#    for x2 in range(N_inducing_points):
#        if COVARIANCE_KERNEL_KX == "periodic":
#            K_fu[x1,x2] = gaussian_periodic_covariance(path[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
#        else:
#            K_fu[x1,x2] = gaussian_NONPERIODIC_covariance(path[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
#K_uf = K_fu.T

# K_gg = K_uu and means inducing points
K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
K_gg_inverse = np.linalg.inv(K_gg)
#K_uu = np.zeros((N_inducing_points,N_inducing_points))
#for x1 in range(N_inducing_points):
#    for x2 in range(N_inducing_points):
#        if COVARIANCE_KERNEL_KX == "periodic":
#            K_uu[x1,x2] = gaussian_periodic_covariance(x_grid_induce[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
#        else:
#            K_uu[x1,x2] = gaussian_NONPERIODIC_covariance(x_grid_induce[x1],x_grid_induce[x2], sigma_f_fit, delta_f_fit)
#K_uu = K_uu + sigma_n/np.sqrt(T) *np.identity(N_inducing_points)
#K_uu += 0.05*np.identity(N_inducing_points)
#K_uu_inverse = np.linalg.inv(K_uu)

#print("Making spatial covariance matrice: Kx crossover beween observations and grid")
# Goes through inducing points
K_g_plotgrid = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)
K_plotgrid_g = K_g_plotgrid.T
#K_u_grid = np.zeros((N_inducing_points,N_plotgridpoints))
#for x1 in range(N_inducing_points):
#    for x2 in range(N_plotgridpoints):
#        if COVARIANCE_KERNEL_KX == "periodic":
#            K_u_grid[x1,x2] = gaussian_periodic_covariance(x_grid_induce[x1],x_grid_for_plotting[x2], sigma_f_fit, delta_f_fit)
#        else:
#            K_u_grid[x1,x2] = gaussian_NONPERIODIC_covariance(x_grid_induce[x1],x_grid_for_plotting[x2], sigma_f_fit, delta_f_fit)
#K_grid_u = K_u_grid.T

# Plot K_g_plotgrid
fig, ax = plt.subplots()
kx_cross_mat = ax.matshow(K_g_plotgrid, cmap=plt.cm.Blues)
fig.colorbar(kx_cross_mat, ax=ax)
plt.title("K_g_plotgrid")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-K_g_plotgrid.png")
print("Making spatial covariance matrice: Kx grid")

K_plotgrid_plotgrid = squared_exponential_covariance(x_grid_for_plotting.reshape((N_plotgridpoints,1)),x_grid_for_plotting.reshape((N_plotgridpoints,1)), sigma_f_fit, delta_f_fit)
#K_grid_grid = np.zeros((N_plotgridpoints,N_plotgridpoints))
#for x1 in range(N_plotgridpoints):
#    for x2 in range(N_plotgridpoints):
#        if COVARIANCE_KERNEL_KX == "periodic":
#            K_grid_grid[x1,x2] = gaussian_periodic_covariance(x_grid_for_plotting[x1],x_grid_for_plotting[x2], sigma_f_fit, delta_f_fit)
#        else:
#            K_grid_grid[x1,x2] = gaussian_NONPERIODIC_covariance(x_grid_for_plotting[x1],x_grid_for_plotting[x2], sigma_f_fit, delta_f_fit)
# 27.03 removing sigma from Kx grid since it will hopefully be taken care of by subtracting less (or fewer inducing points?)
#K_grid_grid += sigma_n*np.identity(N_plotgridpoints) # Here I am adding sigma to the diagonal because it became negative otherwise. 24.03.20

# Plot K_plotgrid_plotgrid
fig, ax = plt.subplots()
kxmat = ax.matshow(K_plotgrid_plotgrid, cmap=plt.cm.Blues)
fig.colorbar(kxmat, ax=ax)
plt.title("K_plotgrid_plotgrid")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-K_plotgrid_plotgrid.png")

Q_gx = np.matmul(np.matmul(K_plotgrid_g, K_gg_inverse), K_gx)
Q_xg = Q_gx.T
#Q_grid_f = np.matmul(np.matmul(K_grid_u, K_uu_inverse), K_uf)
#Q_f_grid = Q_grid_f.T

# Infer mean on the grid
smallinverse = np.linalg.inv(K_gg*sigma_n**2 + np.matmul(K_gx, K_xg))
Q_xx_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_xg, smallinverse), K_gx)
Kxx_times_F = np.matmul(Q_xx_plus_sigma_inverse, F_estimate.T)
mu_posterior = np.matmul(Q_gx, Kxx_times_F) # Here we have Kx crossover. Check what happens if swapped with Q = KKK
#smallinverse = np.linalg.inv(K_uu*sigma_n**2 + np.matmul(K_uf, K_fu))
#Q_ff_plus_sigma_inverse = sigma_n**-2 * np.identity(T) - sigma_n**-2 * np.matmul(np.matmul(K_fu, smallinverse), K_uf)
#pre = np.matmul(Q_ff_plus_sigma_inverse, f_values_observed.T)
#mu_posterior = np.matmul(Q_grid_f, pre) # Here we have Kx crossover. Check what happens if swapped with Q = KKK

# Calculate standard deviations
sigma_posterior = K_plotgrid_plotgrid - np.matmul(Q_gx, np.matmul(Q_xx_plus_sigma_inverse, Q_xg))
#sigma_posterior = K_grid_grid - np.matmul(Q_grid_f, np.matmul(Q_ff_plus_sigma_inverse, Q_f_grid))

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
    plt.plot(x_grid_for_plotting, h_estimate[i,:], color=plt.cm.viridis(0.5), label="Estimated expectation") 
#    plt.plot(x_grid_for_plotting, mu_posterior[i,:], color=plt.cm.viridis(0.5)) 
    plt.title("Expected and average number of spikes, neuron "+str(i)) #spikes
#    plt.title("Neuron "+str(i)+" with "+str(sum(y_spikes[i,:]))+" spikes")
    plt.ylim(ymin=0., ymax=max(1, 1.05*max(observed_mean_spikes_in_bins[i,:]), 1.05*max(h_estimate[i,:])))
    plt.xlabel("X")
    plt.ylabel("Number of spikes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-tuning-"+str(i)+".png")

colors = [plt.cm.viridis(t) for t in np.linspace(0, 1, N)]
plt.figure()
for i in range(N):
    plt.plot(x_grid_for_plotting, observed_mean_spikes_in_bins[i,:], color=colors[i])
#    plt.plot(x_grid_for_plotting, h_estimate[neuron[i,j],:], color=plt.cm.viridis(0.5)) 
    plt.xlabel("X")
    plt.ylabel("Average number of spikes")
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-paral-robust-tuning-collected.png")
#plt.show()
