#Implementing simulations of proxy assets and fund using ABM with correlated Brownian motions
# Cormac Tredoux 2025
#External libraries of numpy, matplotlib, pandas and os called

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def testSim():
    ## Setting working directory 
    os.chdir('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Fixed Weights Hedge')

    ## Variables 
    file = "Combined_data.xlsx"
    Fund = "QQQ"
    ProxyCols = ["QCOM", "RFL", "CSCO" ] 
    weights = np.array([0.3, 0.3, 0.4])

    ## Reading in the historical data 
    data = read_xlsx_file(file)

    ## Extracting 
    price_data = data[ProxyCols].to_numpy()

    ## Calculating volatility and correlations 
    delta = np.diff(price_data, axis=0)  
    sigma_annual = delta.std(ddof=1, axis=0) * np.sqrt(252)  
    corr = np.corrcoef(delta.T)   

    fund_calc = np.dot(price_data, weights)  
    delta_fund = np.diff(fund_calc)
    sigma_annual_fund = delta_fund.std(ddof=1) * np.sqrt(252)  

    ## Parameters
    r = 0.035         
    T_sim = 1200  
    dt = 1/252         
    n = len(ProxyCols)

    # Cholesky
    L_corr = np.linalg.cholesky(corr)

    ## Simulating
    S_sim = np.zeros((T_sim + 1, n))
    S0 = price_data[-1, :]   
    S_sim[0] = S0

    for t in range(T_sim):
        Z = np.random.randn(n)
        dW = L_corr.dot(Z) * np.sqrt(dt)  
        S_sim[t+1] = S_sim[t] + r * S_sim[t] * dt + sigma_annual * dW

    ## Fund trajectories
    F_hist = np.dot(price_data, weights)   
    F_sim = np.dot(S_sim, weights)        

    return F_sim, price_data, S_sim, F_hist, ProxyCols, T_sim, corr, weights, sigma_annual, r, sigma_annual_fund


    

      


#Want to simulate 652 days -> 400 for training and 252 for testing


def sim_year(Proxies, Fund, r, T_sim):
    
    n = Proxies.shape[1]
    dt = 1/504
    
    ## Calculating volatility and correlations for proxies
    delta = np.diff(Proxies, axis=0)  
    sigma_annual = delta.std(ddof=1, axis=0) * np.sqrt(252)  
    corr = np.corrcoef(delta.T)   
    
    # Cholesky
    L_corr = np.linalg.cholesky(corr + 1e-12*np.eye(n))
    
    ## Calculating volatility for fund
    delta_y = np.diff(Fund)
    sigma_fund = delta_y.std(ddof = 1, axis = 0) * np.sqrt(252)

    ## Simulating
    S_sim = np.zeros((T_sim + 1, n))
    S0 = Proxies[-1, :]   
    S_sim[0] = S0
    
    F_sim = np.zeros((T_sim + 1))
    F0 = Fund[-1]
    F_sim[0] = F0

    for t in range(T_sim):
        #Random normal draws
        Zp = np.random.randn(n)
        Zf = np.random.randn()
        #Composing BM with cholesky to correlate
        dWp = L_corr.dot(Zp) * np.sqrt(dt) 
        dWf = Zf*np.sqrt(dt)
        
        S_sim[t+1] = S_sim[t] + r * S_sim[t] * dt + sigma_annual * dWp
        F_sim[t+1] = F_sim[t] + r * F_sim[t] * dt + sigma_fund * dWf
 
    return S_sim.T, F_sim
    