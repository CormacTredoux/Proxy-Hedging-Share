#Implementation of pricing formulas for options on a fund
# Cormac Tredoux 2025
#External libraries of numpy, scipy.stats and os called

import numpy as np
from scipy.stats import norm
import os

## Setting working directory 
os.chdir('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Fixed Weights Hedge')


def Price(K, S_paths, weights, sigma, corr, r, T, t):
    
    tau = T - t 
    
    m = np.dot(weights, S_paths[:,t]) 
    
    #If option has expired simply return the payoff
    if(tau <= 0):
        return max(K - m, 0 ), 0
    
    
    cov = np.outer(sigma, sigma)*corr 
      #Computing covariance
    
    var_com = weights @ cov @ weights 
    
    total_var = ((np.exp(2 * r * tau) -1) / (2 * r))*var_com
    
    d = (K - np.exp(r * tau)*m) / np.sqrt(total_var) 
    
    #Compute price
    Vt = np.exp(-r * tau) * ((K - np.exp(r * tau)*m) * norm.cdf(d)  + np.sqrt(total_var) * norm.pdf(d))
    
    #Compute gradient
    grad = -norm.cdf(d)
    
    return Vt, grad


def Price_F(K, F_paths, sigma, r, T, t):
    
    tau = T - t 
    
    #Current fund value
    
    #If option has expired simply return the payoff
    if(tau <= 0):
        return max(K - F_paths[t], 0 ), 0
    
    mu = F_paths[t]*np.exp(r*tau)
    
    sigma_star_sq = ((sigma**2)/(2*r))*np.exp(2*r*tau)*(1-np.exp(-2*r*tau))
    
    H = (K - mu)/np.sqrt(sigma_star_sq)
    
    #Compute price
    Vt = np.sqrt(sigma_star_sq) * np.exp(-r * tau) * (H * norm.cdf(H)  + norm.pdf(H))
    
    #Compute gradient
    grad = -norm.cdf(H)
    
    return Vt, grad