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