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
plt.rc('image', cmap='viridis') #, linewidth=1)
from scipy import optimize
numpy.random.seed(13)
from sklearn.linear_model import LinearRegression

# Simulated data
# 20 iterations
# I was working on the computer in the same time so they're not perfect
Nseeds = 5

T_values = [10,250,500,750,1000] 
times = {
        '10' : [2.8322057723999023, 3.7169768810272217, 3.5322799682617188, 3.995462417602539, 3.632303237915039, 5.046312570571899, 4.436202764511108, 4.432348728179932, 4.222238302230835, 4.541436672210693], 
        '250' : [49.890037059783936, 47.54969763755798, 48.47557592391968, 45.93554639816284, 51.372334241867065, 64.90720248222351, 48.01697492599487, 45.96428036689758, 44.372485637664795, 44.01561784744263],
        '500' : [129.28476738929749, 122.03907465934753, 104.54554295539856, 116.40743231773376, 153.46324801445007, 131.97159624099731, 97.58699417114258, 101.64296412467957, 100.36379289627075, 75.68742060661316],
        '750' : [154.3104064464569, 223.40830183029175, 162.83168840408325, 161.83721804618835, 195.2042293548584, ],
        '1000' : [188.4972276687622, 241.1461215019226, 212.54603791236877, 199.13066482543945, 226.39088988304138]
    }
mean_values = [np.mean(times[str(T)]) for T in T_values]

# Least squares polynomial fit
x = np.array([10,10,10,10,10,250,250,250,250,250,500,500,500,500,500,750,750,750,750,750,1000,1000,1000,1000,1000]).reshape((-1, 1))
y = np.array([2.8322057723999023, 3.7169768810272217, 3.5322799682617188, 3.995462417602539, 3.632303237915039, 49.890037059783936, 47.54969763755798, 48.47557592391968, 45.93554639816284, 51.372334241867065, 129.28476738929749, 122.03907465934753, 104.54554295539856, 116.40743231773376, 153.46324801445007, 154.3104064464569, 223.40830183029175, 162.83168840408325, 161.83721804618835, 195.2042293548584, 188.4972276687622, 241.1461215019226, 212.54603791236877, 199.13066482543945, 226.39088988304138])
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred_self = model.predict(x)
print("RMSE of line fit vs data:", np.sqrt(np.sum((y-y_pred_self)**2)/len(x)))
y_pred = model.predict(np.array(T_values).reshape((-1, 1)))
print('predicted response:', y_pred, sep='\n')

plt.figure()
plt.title("Runtime")
plt.xlabel("Time bins")
plt.ylabel("Runtime (s)")
plt.yscale("log")
plt.xscale("log")
for T_index in range(len(T_values)):
    T = T_values[T_index]
    #plt.errorbar(x=T, y=np.mean(times[str(T)]), yerr=(np.std(times[str(T)])), fmt=".", markersize=5, capsize=2, linewidth=1, label="T="+str(T), color = plt.cm.viridis(0.8*T_index/len(times)))
    plt.scatter(x,y, s=3, label="T = "+ str(T)) #, lw=0)
plt.plot(np.array(T_values).reshape((-1, 1)), y_pred, "-", label="Linear regression fit", color=plt.cm.viridis(0.3))
plt.legend()
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-timeuse.png")
plt.show()
exit()




#################### LOG SCALE #########################################
print("Entering log landscape!!!!!!!!")
# Simulated data
# 20 iterations
Nseeds = 5

T_values = [10,32,100,316,1000] #500

times = {
    '10' : [2.8322057723999023, 3.7169768810272217, 3.5322799682617188, 3.995462417602539, 3.632303237915039], 
    '32' : [10.206804037094116, 10.890196561813354, 12.050742864608765, 11.517948627471924, 8.884598016738892],
    '100' : [21.48021125793457, 25.170532703399658, 43.63818049430847, 20.60857915878296, 19.43113684654236], 
    '316' : [62.4989378452301, 69.12636995315552, 68.26071214675903, 65.95395874977112, 66.80253434181213],
    '500' : [129.28476738929749, 122.03907465934753, 104.54554295539856, 116.40743231773376, 153.46324801445007],
    '1000' : [188.4972276687622, 241.1461215019226, 212.54603791236877, 199.13066482543945, 226.39088988304138]
    }
mean_values = [np.mean(times[str(T)]) for T in T_values]

# Least squares polynomial fit
x = np.array([10,10,10,10,10,32,32,32,32,32,100,100,100,100,100,316,316,316,316,316,1000,1000,1000,1000,1000]).reshape((-1, 1))
y = np.array([2.8322057723999023, 3.7169768810272217, 3.5322799682617188, 3.995462417602539, 3.632303237915039, 10.206804037094116, 10.890196561813354, 12.050742864608765, 11.517948627471924, 8.884598016738892, 21.48021125793457, 25.170532703399658, 43.63818049430847, 20.60857915878296, 19.43113684654236, 62.4989378452301, 69.12636995315552, 68.26071214675903, 65.95395874977112, 66.80253434181213, 188.4972276687622, 241.1461215019226, 212.54603791236877, 199.13066482543945, 226.39088988304138])
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(np.array(T_values).reshape((-1, 1)))
print('predicted response:', y_pred, sep='\n')
#print("RMSE of line fit vs data:")

plt.figure()
plt.title("Time use")
plt.xlabel("Time bins")
plt.ylabel("Time use (s)")
plt.yscale("log")
plt.xscale("log")
for T_index in range(len(T_values)):
    T = T_values[T_index]
    #plt.errorbar(x=T, y=np.mean(times[str(T)]), yerr=(np.std(times[str(T)])), fmt=".", markersize=5, capsize=2, linewidth=1, label="T="+str(T), color = plt.cm.viridis(0.8*T_index/len(times)))
    plt.scatter(x,y, s=3, label="T = "+ str(T)) #, lw=0)
#plt.plot(T_values, model.intercept_ + [model.coef_ * ttt for ttt in T_values], '-', linewidth=1, label="Least squares fit")
#plt.plot(np.log(T_values), -1 + np.log(model.intercept_ + [model.coef_ * ttt for ttt in T_values]), '-', linewidth=1, label="Least squares fit")
plt.plot(np.array(T_values).reshape((-1, 1)), y_pred, "-", label="Linear regression fit")
plt.legend()
plt.tight_layout()
plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-timeuse.png")
plt.show()

