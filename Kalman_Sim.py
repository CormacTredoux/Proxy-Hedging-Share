# Implementation of Kalman Filter hedging on simulated paths 
# Cormac Tredoux 2025
# External libraries of numpy, matplotlib and pandas called

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from KalmanFilter import *
from utils import *
from Hedging import *
from Pricing import * 
from Simulations import *
import random

np.random.seed(941)
random.seed(941) 
# Reading in data
file = "Combined_data.xlsx"
Fund_name = "QQQ"
ProxyCols = ["QCOM", "RFL", "CSCO"]  
NumProxies = len(ProxyCols)
data = read_xlsx_file(file)

bach_train = 100

Proxies = data[ProxyCols].to_numpy()
Proxies_train = Proxies[300: 300 + bach_train, :]
Fund = data[Fund_name].to_numpy()
Fund_train = Fund[300:300 + bach_train]

# risk free rate 
r = 0.05

# Number of simulations
n_sim = 10000

finals = np.zeros(n_sim)
avte = np.zeros(n_sim)

for i in range(n_sim):
    # Number of days per simulation
    T_sim = 252

    # Simulating
    X, Y = sim_year(Proxies_train, Fund_train, r, T_sim)
    
    
    testing  = 252                               
    training = 400                               
    start_test  = 600                           
    start_train = start_test - training          
   

    # Combining simulated data with historical data for training
    X = np.vstack([Proxies[start_train:start_test, :], X.T]).T
    Y = np.concatenate([Fund[start_train:start_test], Y])

    # Just creates a list with elements of Y, transposes and converts to numpy
    Y = np.array([[y for y in Y.T]])

    ProxyNames = np.array(ProxyCols)

    ## Random Set-up
    # Number of assets
    d = X.shape[0]
    # nobs
    n = X.shape[1]
    # All proxies at time k...
    xs = [X[:, k] for k in range(0, n)]
    # Same vibes but (1,) where each scalar wrapped into an array
    ys = [Y[:, k] for k in range(0, n)]

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    load_dir = os.path.join(parent_dir, "Saved_States")
    params = np.loadtxt(os.path.join(load_dir, "MLE_params.txt"))

    state_0, P_0, Q, R = set_params(params, d)
    _, states, errs, Ps, Ss, _ = Kalman_run(xs, ys, state_0, P_0, Q, R, d)

    weights = [states[i].flatten() for i in range(training, training + testing)]

    FundVals = Y[0]

    
    FundRep = np.array([
        np.dot(weights[i - training], X[:, i])
        for i in range(training, training + testing)
    ])

    FundToPlot = FundVals[training: training + testing]
    FundRepToPlot = FundRep
    time_axis = np.arange(training, training + testing)

    ### Hedging ###
    year = 252
    expiry_date = T = testing

    t0       = training
    t_expiry = training + testing - 1

    # Strike price
    end_training_abs = start_test - 1              
    end_test_abs     = end_training_abs + testing  
    tau = (end_test_abs - end_training_abs) / year  
    Strike = float(Fund[end_training_abs]) * np.exp(r * tau)
    
    

    X_train = X[:, :training]
    X_test  = X[:, training: training + testing]
    Y_train = Y[0, :training]
    Y_test  = Y[0, training: training + testing]

    # Computing volatility and correlations 
    delta = np.diff(X_train, axis=1) 
    sigma_annual = delta.std(axis=1, ddof=1) * np.sqrt(252) 
    corr = np.corrcoef(delta, rowvar=True)     

    delta_y = np.diff(Y_train)
    sigma_fund = delta_y.std() * np.sqrt(252)

    tracking_errors, Hedge, Vp, grads = ProxyHedge_self(
        Strike, X_test, weights, sigma_annual, corr, r, T, Price, sigma_fund, Y_test
    )

    tracking_errors2, _, V, _ = Hedge_F(
        Strike, Y_test, sigma_fund, r, T, Price_F
    )
    
    finals[i] = (Hedge[testing-1] - Hedge[testing - 2]) - (V[testing-1] - V[testing - 2])
    
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
plt.show()
plt.savefig("kalman_plot.png", dpi=200) 


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

finals = pd.DataFrame(finals)
finals.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/Kalman_sim_hedge.xlsx', index=False, header=False)

avte = pd.DataFrame(avte)
avte.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/Kalman_sim_hedge_avte.xlsx', index=False, header=False)
