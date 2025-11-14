#Implementation of OLS regression to compute asset weights
# Cormac Tredoux 2025
#External libraries of numpy, pandas and sklearn called
# Importantly uses LinearRegression from sklearn, this is one of the few algorithms that was not coded from first principles for this project.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import read_xlsx_file



#Function that fits OLS regression to given testing period (in days) and returns the weights

def fitOLSweights(assets, fund): 


    n = assets.shape[1]

    # Train
    X_train = assets
    Y_train = fund

    #Fitting the regression model
    OLS_model = LinearRegression(fit_intercept=False)
    OLS_model.fit(X_train, Y_train)
    weights = OLS_model.coef_
   
    
    
    #Calibrating volatility and correlation of training data 
    delta = np.diff(X_train, axis=0)  

    sigma = delta.std(axis=0, ddof=1) * np.sqrt(252)  
    corr = np.corrcoef(delta, rowvar=False)        
    
    delta_y = np.diff(Y_train)
    sigma_fund = delta_y.std()*np.sqrt(252)
    


    
    return weights, sigma, corr, sigma_fund