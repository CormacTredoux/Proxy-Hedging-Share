# Implementation of Maximum Likelihood Estimation for Kalman Filter parameters
# Cormac Tredoux 2025
#External libraries of numpy and scipy called

import numpy as np
from scipy.optimize import minimize
from KalmanFilter import *
from utils import *

#Function to be optimised 
def neg_loglik(params, xs, ys, d):
    #Unpack the parameters
    state0, P0, Q, R = set_params(params, d)
    #Extratc log likelihood 
    loglik, _, _, _,_,_ = Kalman_run(xs, ys, state0, P0, Q, R, d)
    return -loglik



# Reading in data

file = "Combined_data.xlsx"
Fund = "QQQ"
ProxyCols = ["QCOM", "RFL", "CSCO"]  
NumProxies = len(ProxyCols)
data = read_xlsx_file(file)


# Window configuration

train_start = 200  
training    = 400   
test_start  = train_start + training  
testing     = 252          


# Prepare arrays

dates = data["Date"]
X = data[ProxyCols].to_numpy().T
Y = data[Fund]
Y = np.array([[y for y in Y.to_numpy().T]])   

ProxyNames = np.array(ProxyCols)

# Number of assets and nobs
d = X.shape[0]
n = X.shape[1]

# Lists of observations across full sample (Kalman will run across all)
xs = [X[:, k] for k in range(n)]             
ys = [Y[:, k] for k in range(n)]            


#Initialising parameters
init_params = np.ones(int(d + d*(d+1)//2 + d*(d+1)//2 + 1))

res = minimize(neg_loglik, init_params, args=(xs[train_start:train_start+training], ys[train_start:train_start+training], d))

parent_dir = os.path.dirname(os.path.dirname(__file__))  
save_dir = os.path.join(parent_dir, "Saved_States")
os.makedirs(save_dir, exist_ok=True)

np.savetxt(os.path.join(save_dir, "MLE_params.txt"), res.x)

