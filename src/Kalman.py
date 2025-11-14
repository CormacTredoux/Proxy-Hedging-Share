# Implementation of Kalman Filter hedging on historical data
# Cormac Tredoux 2025
# External libraries of numpy, matplotlib and pandas, statsmodels, scipy, datetime and os called

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from KalmanFilter import *
from utils import *
from Hedging import *
from Pricing import * 
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


# Reading in data

file = "Combined_data.xlsx"
Fund = "QQQ"
ProxyCols = ["QCOM", "RFL", "CSCO"]  
NumProxies = len(ProxyCols)
data = read_xlsx_file(file)


# Window configuration

train_start = 200          
training    = 400          
test_start  = train_start + training   # 520
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

# Setting up observations for Kalman
xs = [X[:, k] for k in range(n)]           
ys = [Y[:, k] for k in range(n)]             


# Kalman filter

parent_dir = os.path.dirname(os.path.dirname(__file__))
load_dir = os.path.join(parent_dir, "Saved_States")
params = np.loadtxt(os.path.join(load_dir, "MLE_params.txt"))


state_0, P_0, Q, R = set_params(params, d)
print(state_0)
print(P_0)
print(Q)
print(R)
_, states, errs, Ps,Ss, innovs = Kalman_run(xs, ys, state_0, P_0, Q, R, d)

SAVE_DIR = '/Users/cormactredoux/Documents/Uni/Proxy Hedging/Write-up/Figures'
# Checking normality 
weights = [states[i].flatten() for i in range(test_start, test_start + testing)]

#Standardising residuals (forecast errors)
z = np.array(innovs)/np.sqrt(np.array(Ss))
z = z[train_start:train_start+training]  



## QQ Plot
fig = plt.figure(figsize=(6, 6))
stats.probplot(z, dist="norm", plot=plt)
plt.xlabel("Theoretical Quantiles", fontsize=11)
plt.ylabel("Sample Quantiles", fontsize=11)
plt.grid(alpha=0.3, linestyle="--")
plt.gca().get_lines()[1].set_color("red")
plt.gca().get_lines()[1].set_linewidth(1)
plt.gca().set_title("")
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "QQ.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# Histogram + normal overlay
fig = plt.figure(figsize=(6, 6))
plt.hist(z, bins=25, density=True, alpha=0.55, color="steelblue", edgecolor="white")
x = np.linspace(-4, 4, 300)
plt.plot(x, stats.norm.pdf(x), "r-", lw=1)
plt.xlabel("Standardised Innovation", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.grid(alpha=0.3, linestyle="--")
plt.title("")
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "Innovations_hist.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ACF
fig, ax = plt.subplots(figsize=(6, 6))
plot_acf(z, lags=30, zero=True, ax=ax)  
ax.set_xlabel("Lag", fontsize=11)
ax.set_ylabel("Autocorrelation", fontsize=11)
ax.grid(alpha=0.3, linestyle="--")
ax.set_title("")
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "ACF.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)






# Extract and cast in normal format
weights = [states[i].flatten() for i in range(test_start, test_start + testing)]



# Replication

FundVals = Y[0]

FundRep = np.array([
    np.dot(weights[i - test_start], X[:, i])
    for i in range(test_start, test_start + testing)
])

FundToPlot = FundVals[test_start:test_start + testing]
FundRepToPlot = FundRep
time_axis = np.arange(test_start, test_start + testing)

plt.figure(figsize=(10, 6))
plt.plot(time_axis, FundToPlot, label="QQQ")
plt.plot(time_axis, FundRepToPlot, label="Replication Portfolio")
plt.title("Fund Replication for QQQ")
plt.legend()

# Hedging setup
r = 0.05
year = 252
T = testing  

# Key indices
end_training = train_start + training - 1   
end_test     = test_start  + testing  - 1   

# Time from end of training to end of test (in years)
tau = (end_test - end_training) / year

# Strike: value at end of training, appreciated to end of test at rate r
Strike = float(FundVals[end_training]) * np.exp(r * tau)



# Train/test splits per new window
X_train = X[:, train_start: train_start + training]
X_test  = X[:, test_start : test_start  + testing]
Y_train = Y[0, train_start: train_start + training]
Y_test  = Y[0, test_start : test_start  + testing]


# Volatility & correlations

delta = np.diff(X_train, axis=1)                            
sigma_annual = delta.std(axis=1, ddof=1) * np.sqrt(252)      
corr = np.corrcoef(delta, rowvar=True)                      

delta_y = np.diff(Y_train)
sigma_fund = delta_y.std() * np.sqrt(252)


# Hedging


tracking_errors, Hedge, Vp, grads = ProxyHedge_self(
    Strike, X_test, weights, sigma_annual, corr, r, T, Price, sigma_fund, Y_test
)

tracking_errors2, _, V, _ = Hedge_F(
    Strike, Y_test, sigma_fund, r, T, Price_F
)


# Plot with actual dates for the new window

if 'Date' in data.columns:
    date_series = pd.to_datetime(data['Date'].iloc[test_start:test_start + testing], errors='coerce')
    date_range  = np.array(date_series.dt.to_pydatetime())
else:
    start_date_fallback = datetime(2000, 1, 1)
    date_range = pd.bdate_range(start=start_date_fallback, periods=testing, freq='B').to_pydatetime()

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(date_range, V[:testing],        label='Liability $V_t$',                      linewidth=2)
ax.plot(date_range, Hedge[:testing],    label='Hedge $H_t$',                          linewidth=2)
ax.plot(date_range, tracking_errors[:testing], label='Tracking Error ($\\Delta H_t - \\Delta V_t$)',
        linestyle='--', linewidth=2)

ax.set_xlim(date_range[0], date_range[-1] + timedelta(days=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%b'))
ax.minorticks_off()

ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Kalman Filter Hedge performance on historical backtest', fontsize=12, pad=20)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.setp(plt.gca().lines, linewidth=1.0)
plt.show()


# Export results

V_df = pd.DataFrame(V)
V_df.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/Liability_test.xlsx', index=False, header=False)
Hedge_df = pd.DataFrame(Hedge)
Hedge_df.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/Kalman_test_hedge.xlsx', index=False, header=False)

FundRep_df = pd.DataFrame(FundRep)
FundRep_df.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/Kalman_test_fundrep.xlsx', index=False, header=False)
grads_df = pd.DataFrame(grads)
grads_df.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/Kalman_deltas.xlsx', index=False, header=False)

weights_df = pd.DataFrame(weights)
weights_df.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/Kalman_test_weights.xlsx', index=False, header=False)