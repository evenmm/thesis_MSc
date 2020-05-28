#### Eigenvalue testing
#path = np.ones(T)
K_xg = squared_exponential_covariance(path.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
K_gx = K_xg.T
K_gg_inv = np.linalg.inv(K_gg)
#print("np.matmul(K_gx,K_xg)\n",np.matmul(K_gx,K_xg))
#print("np.matmul(K_xg,K_gx)\n",np.matmul(K_xg,K_gx))

fig, ax = plt.subplots()
foo_mat = ax.matshow(K_gx) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("K_gx")
plt.tight_layout()
fig, ax = plt.subplots()
foo_mat = ax.matshow(np.matmul(K_gx, K_xg)) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("np.matmul(K_gx, K_xg)")
plt.tight_layout()
fig, ax = plt.subplots()
foo_mat = ax.matshow(np.matmul(K_xg,K_gx)) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("np.matmul(K_xg,K_gx)")
plt.tight_layout()

A_full = np.matmul(K_xg, np.matmul(K_gg_inv, K_gx))*sigma_n**(-2) + np.identity(T)
A_reduced = np.matmul(K_gg_inv, np.matmul(K_gx, K_xg))*sigma_n**(-2) + np.identity(N_inducing_points)
print("\ndet(A full)",np.prod(np.linalg.eigvals(A_full)))
print("det(A_reduced)",np.prod(np.linalg.eigvals(A_reduced)))

fig, ax = plt.subplots()
foo_mat = ax.matshow(np.matmul(K_xg, np.matmul(K_gg_inv, K_gx))*sigma_n**(-2)) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("A full minus I")
plt.tight_layout()

fig, ax = plt.subplots()
foo_mat = ax.matshow(np.matmul(K_gg_inv, np.matmul(K_gx, K_xg))*sigma_n**(-2)) #cmap=plt.cm.Blues
fig.colorbar(foo_mat, ax=ax)
plt.title("A_reduced minus I")
plt.tight_layout()

A_nextline = np.matmul(K_gx, K_xg)*sigma_n**(-2) + K_gg
print("\nA_nextline = np.matmul(K_gx, K_xg)*sigma_n**(-2) + K_gg")
print("det(A nextline)",np.prod(np.linalg.eigvals(A_nextline)))
print("det(K_gx * K_xg)*sigma_n**(-2)\n",np.prod(np.linalg.eigvals(np.matmul(K_gx, K_xg)*sigma_n**(-2))))
print("det(K_gg)",np.prod(np.linalg.eigvals(K_gg)))
print("det(A_nextline)",np.prod(np.linalg.eigvals(A_nextline)))

print("det(A_nextline)/det(K_gg)", np.prod(np.linalg.eigvals(A_nextline)) / np.prod(np.linalg.eigvals(K_gg)))

print("\nEigenvalues of A full\n",np.linalg.eigvals(A_full))
print("Eigenvalues of A reduced\n",np.linalg.eigvals(A_reduced))
print("Eigenvalues of A full minus I\n",np.linalg.eigvals(np.matmul(K_xg, np.matmul(K_gg_inv, K_gx))*sigma_n**(-2)))
print("Eigenvalues of A reduced\n",np.linalg.eigvals(np.matmul(K_gg_inv, np.matmul(K_gx, K_xg))*sigma_n**(-2)))

print("\nEigenvalues of K_gg^-1", np.linalg.eigvals(K_gg_inv))
print("Eigenvalues of K_gx*K_xg", np.linalg.eigvals(np.matmul(K_gx, K_xg)))
print("Eigenvalues of K_xg*K_gx\n", np.linalg.eigvals(np.matmul(K_xg, K_gx)))

print("det(np.matmul(K_gx, K_xg))",np.prod(np.linalg.eigvals(np.matmul(K_gx, K_xg))))
print("det(K_gg)\n",np.prod(np.linalg.eigvals(K_gg)))
print("det(I_T)",np.prod(np.linalg.eigvals(np.identity(T))))
print("det(sigma^2 I_T)",np.prod(np.linalg.eigvals(sigma_n**2 * np.identity(T))))
print("det(sigma^2 I_N_ind)",np.prod(np.linalg.eigvals(sigma_n**2 * np.identity(N_inducing_points))))

print("\nEigenvalues of K_gg\n", np.linalg.eigvals(K_gg))
print("Eigenvalues of K_gg plain\n", np.linalg.eigvals(K_gg_plain))
print("det(K_gg_plain)",np.prod(np.linalg.eigvals(K_gg_plain)))
plt.show()

