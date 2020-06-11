    
    # Plotting all initial F:
    ## Plot true f
    fig, ax = plt.subplots(figsize=(8,2))
    foo_mat = ax.matshow(true_f) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("True f")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-i-true-f.png")

    ## Plot initial f
    fig, ax = plt.subplots(figsize=(8,2))
    foo_mat = ax.matshow(F_initial) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("Initial F")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-i-initial-f.png")

    ## Plot initial f - variant
    fig, ax = plt.subplots(figsize=(8,2))
    foo_mat = ax.matshow(np.log(y_spikes + 0.0008)) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("Initial F")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-i-initial-f-variant.png")

    # Plot y spikes
    fig, ax = plt.subplots(figsize=(8,2)) # figsize=(8,1)
    foo_mat = ax.matshow(y_spikes) #cmap=plt.cm.Blues
    plt.title("Spikes")
    fig.colorbar(foo_mat, ax=ax)
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-y-spikes.png")

    # rmse for basic bitch initial:
    print("Basic bitch:",np.sqrt(np.mean(np.sum((true_f - F_initial)**2))))

    ## Plot variant of initial f : GLM style
    epsilonnn = 0.0008

    def rmsebetweentrueFandInitial(epsilon):
        return np.sqrt(np.mean(np.sum((true_f - (np.log(y_spikes + epsilon)))**2)))
    epsarray = np.linspace(0.0005, 0.0015,10) #[0.1,0.01,0.001,0.0001]
    print(epsarray)
    for i in range(len(epsarray)):
        print(rmsebetweentrueFandInitial(epsarray[i]))

    fig, ax = plt.subplots(figsize=(8,2))
    foo_mat = ax.matshow(np.log(y_spikes + epsilonnn)) #cmap=plt.cm.Blues
    fig.colorbar(foo_mat, ax=ax)
    plt.title("Initial f")
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-initial-f-variant.png")

    F_count = np.ndarray.flatten(true_f)
    print(F_count)
    print("Minmax Fcount", min(F_count), max(F_count))
    plt.figure()
    plt.hist(F_count, bins=7, log=True, color=plt.cm.viridis(0.3))
    plt.ylabel("Number of bins")
    plt.xlabel("Spike count")
    plt.title("True F histogram")
    #plt.xticks(range(int(min(F_count)),int(max(F_count)),1))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-initialcheck-F-true-histogram-log.png")

    ### Histograms ###
    # Sqrtstart
    F_count = np.ndarray.flatten(F_initial)
    print(min(F_count), max(F_count))
    plt.figure()
    plt.hist(F_count, bins=7, log=True, color=plt.cm.viridis(0.3))
    plt.ylabel("Number of bins")
    plt.xlabel("Spike count")
    plt.title("True F histogram")
    #plt.xticks(range(int(min(F_count)),int(max(F_count)),1))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-initialcheck-F-initial-histogram-log.png")

    # Variant
    F_count = np.ndarray.flatten(np.log(y_spikes + 0.10008))
    print(min(F_count), max(F_count))
    plt.figure()
    plt.hist(F_count, bins=7, log=True, color=plt.cm.viridis(0.3))
    plt.ylabel("Number of bins")
    plt.xlabel("Spike count")
    plt.title("True F histogram")
    #plt.xticks(range(int(min(F_count)),int(max(F_count)),1))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-initialcheck-F-initial-variant-histogram-log.png")

    # Spikes
    spike_count = np.ndarray.flatten(y_spikes)
    plt.figure()
    plt.hist(spike_count, bins=np.arange(0,int(max(spike_count))+1)-0.5, log=True, color=plt.cm.viridis(0.3))
    plt.ylabel("Number of bins")
    plt.xlabel("Spike count")
    plt.title("Spike histogram")
    plt.xticks(range(0,int(max(spike_count)),1))
    plt.tight_layout()
    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-simulated-em-initialcheck-spike-histogram-log.png")
    #exit()
