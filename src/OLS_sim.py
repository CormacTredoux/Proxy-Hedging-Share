#Implements OLS hedging on simulated paths 
#Cormac Tredoux 2025
#External libraries of numpy, matplotlib, random, pandas and os called

from OLS_weights import fitOLSweights
from Hedging import *
from Pricing import Price
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
from Simulations import *
import random
import pandas as pd  

np.random.seed(941)
random.seed(941) 

#Reading in data
file = "Combined_data.xlsx"
Fund_name = "QQQ"
ProxyCols = ["QCOM", "RFL", "CSCO"]  
NumProxies = len(ProxyCols)
data = read_xlsx_file(file)

bach_train = 100

Proxies = data[ProxyCols].to_numpy()
Proxies_train = Proxies[300:300 + bach_train, :]
Fund = data[Fund_name].to_numpy()
Fund_train = Fund[300: 300 + bach_train]

# risk free rate 
r = 0.05


training    = 400            
testing     = 252            
start_test  = 600            
start_train = start_test - training   

# Number of simulations
n_sim = 10000

finals = np.zeros(n_sim)
avte = np.zeros(n_sim)

for i in range(n_sim):
    # Number of days per simulation
    T_sim = 252

    #Simulated paths
    X, Y = sim_year(Proxies_train, Fund_train, r, T_sim)
    
   
    X = np.vstack([Proxies[start_train:start_test, :], X.T]).T
    Y = np.concatenate([Fund[start_train:start_test], Y])


    # Splitting on testing and training 
    X_train = X.T
    X_train = X_train[:training, :]
    Y_train = Y
    Y_train = Y_train[:training]

    # Extracting data
    weights, sigma_annual, corr, sigma_fund = fitOLSweights(X_train, Y_train)

    # Strike calculation 
    year = 252
    expiry_date = T = testing  

    t0       = training
    t_expiry = training + testing - 1

    end_training_abs = start_test - 1               
    end_test_abs     = end_training_abs + testing   
    tau = (end_test_abs - end_training_abs) / year  

    Strike = float(Fund[end_training_abs]) * np.exp(r * tau)

    ##### Hedging #####
    X_test = X.T
    X_test = X_test[training:training + testing, :].T
    Y_test = Y
    Y_test = Y_test[training:training + testing].T

    
    delta_y = np.diff(Y_train)
    sigma_fund = delta_y.std() * np.sqrt(252)

    tracking_errors, Hedge, Vp, grads = ProxyHedge_self(
        Strike, X_test, weights, sigma_annual, corr, r, T, Price, sigma_fund, Y_test
    )
    tracking_errors2, _, V, _ = Hedge_F(Strike, Y_test, sigma_fund, r, T, Price_F)
    
    finals[i] = Hedge[testing - 1] - Hedge[testing - 2] - (V[testing - 1] - V[testing - 2])
    avte[i] = np.mean(np.diff(Hedge) - np.diff(V))


mean, std = finals.mean(), finals.std()
x_min, x_max = mean - 3*std, mean + 3*std


bins = 50 
plt.figure(figsize=(6,6))
plt.hist(finals, bins=np.linspace(x_min, x_max, bins), edgecolor='black', alpha=0.7, density=True)

plt.xlabel("Tracking Error")
plt.ylabel("Density")
plt.title("Distribution of Final Tracking Errors on Simulated Paths")
plt.grid(True, linestyle='--', alpha=0.6)


plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f"Mean={mean:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("OLS_plot.png", dpi=200) 

finals = pd.DataFrame(finals)
finals.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/OLS_sim_hedge.xlsx', index=False, header=False)
plt.show()


mean, std = avte.mean(), avte.std()
x_min, x_max = mean - 3*std, mean + 3*std


bins = 50  
plt.figure(figsize=(6,6))
plt.hist(avte, bins=np.linspace(x_min, x_max, bins), edgecolor='black', alpha=0.7, density=True)

plt.xlabel("Average Tracking Error")
plt.ylabel("Density")
plt.title("Distribution of Average Tracking Errors on Simulated Paths")
plt.grid(True, linestyle='--', alpha=0.6)


plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f"Mean={mean:.2f}")
plt.legend()
plt.tight_layout()

plt.show()

avte = pd.DataFrame(avte)
avte.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/OLS_sim_hedge_avte.xlsx', index=False, header=False)
