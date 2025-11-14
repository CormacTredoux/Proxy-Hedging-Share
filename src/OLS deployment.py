#Implementation of OLS hedging on historical data
# Cormac Tredoux 2025
# External libraries of numpy, matplotlib, datetime, pandas and os called

from OLS_weights import fitOLSweights
from Hedging import *
from Pricing import Price
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd

# Risk free rate 
r = 0.05

# Parameters for OLS weight fitting 
file = "Combined_data.xlsx"
Fund_name = "QQQ"
ProxyCols = ["QCOM", "RFL", "CSCO"]  
NumProxies = len(ProxyCols)

# Extracting from file
data = read_xlsx_file(file)

#Window configuration
training    = 400           
testing     = 252           
start_test  = 600           
start_train = start_test - training  


# Splitting on testing and training 
assets_train = data[ProxyCols].to_numpy()
assets_train = assets_train[start_train:start_test, :]
fund_train   = data[Fund_name].to_numpy()
fund_train   = fund_train[start_train:start_test]

# Extracting data
ols_weights, sigma_annual, corr, sigma_fund = fitOLSweights(assets_train, fund_train)

# Strike calculation
year        = 252
expiry_date = T = testing  
t0          = training
t_expiry    = training + testing - 1
tau         = (testing - 1) / year 

##### Hedging #####
fund_train   = fund_train.T
assets_train = assets_train.T

assets = data[ProxyCols].to_numpy()
assets = assets[start_test:start_test + testing, :].T
fund   = data[Fund_name].to_numpy()
fund   = fund[start_test:start_test + testing].T

#Strike
end_training_abs = start_test - 1
end_test_abs     = end_training_abs + testing
Strike = np.exp(r * ((end_test_abs - end_training_abs) / year)) * data[Fund_name].to_numpy()[end_training_abs]

#Strike (ctd)
Strike = np.exp(-r * tau) * fund[testing - 1]
Strike = 240
Strike = float(fund_train[-1]) * np.exp(r * tau)

delta_y    = np.diff(fund_train)
sigma_fund = delta_y.std() * np.sqrt(252)

print("OLS weights shape/type:", np.asarray(ols_weights).shape, type(ols_weights))
print("First 3 OLS weights:", np.asarray(ols_weights).ravel()[:3])



tracking_errors, Hedge, Vp, grads = ProxyHedge_self(Strike, assets, ols_weights, sigma_annual, corr, r, T, Price, sigma_fund, fund)
tracking_errors2, _, V, _        = Hedge_F(Strike, fund, sigma_fund, r, T, Price_F)



#Time axis
time_axis = np.arange(testing)

safe_weights = np.nan_to_num(ols_weights, nan=0.0, posinf=0.0, neginf=0.0)
assets       = np.asarray(assets, dtype=np.float64)
assets       = np.ascontiguousarray(assets, dtype=np.float64)
safe_weights = np.asarray(safe_weights, dtype=np.float64).ravel()

print("assets shape:", assets.shape, "strides:", assets.strides, "dtype:", assets.dtype)
print("weights shape:", safe_weights.shape, "strides:", safe_weights.strides, "dtype:", safe_weights.dtype)

# Recovering fitted fund values
proxy_values = safe_weights @ assets


plt.figure(figsize=(10, 6))
plt.plot(time_axis, fund[:testing],         label='Actual Fund $F_t$',                 color='blue')
plt.plot(time_axis, proxy_values[:testing], label='Proxy Portfolio ($X_t \\cdot w$)',  color='orange')
plt.xlabel('Time (days)')
plt.ylabel('Value')
plt.title('Proxy Quality: Actual Fund vs Proxy Portfolio')
plt.legend()
plt.grid(True)
plt.tight_layout()

if 'Date' in data.columns:
    date_series = pd.to_datetime(data['Date'].iloc[start_test:start_test + testing], errors='coerce')
    date_range  = np.array(date_series.dt.to_pydatetime())
else:

    start_date_fallback = datetime(2022, 5, 11)
    date_range          = pd.bdate_range(start=start_date_fallback, periods=testing, freq='B').to_pydatetime()

start_date     = date_range[0]
end_date_exact = date_range[-1]

start_date = datetime(2022, 9, 6)
end_target = datetime(2023, 9, 7)



mask = (date_range >= start_date) & (date_range <= end_target)
date_range_plot = date_range[mask]
V_plot          = V[:testing][mask]
Hedge_plot      = Hedge[:testing][mask]
tracking_plot   = tracking_errors[:testing][mask]
end_date_exact  = end_target  


pad_days = 7
end_limit = end_date_exact + timedelta(days=pad_days)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(date_range_plot, V_plot,        label='Liability $V_t$',                       linewidth=2)
ax.plot(date_range_plot, Hedge_plot,    label='Hedge $H_t$',                           linewidth=2)
ax.plot(date_range_plot, tracking_plot, label='Tracking Error ($\\Delta H_t - \\Delta V_t$)',
        linestyle='--', linewidth=2)


ax.set_xlim(start_date, end_limit)

def first_trading_on_or_after(dts, target):
    for d in dts:
        if d >= target:
            return d
    return dts[-1]


year_boundary = first_trading_on_or_after(date_range_plot, datetime(2023, 1, 1))

desired_ticks = [
    start_date,
    year_boundary,
    datetime(2022, 11, 17),
    datetime(2023, 2, 17),
    datetime(2023, 5, 17),
    end_date_exact
]
tick_dates = [d for d in desired_ticks if start_date <= d <= end_limit]
ax.set_xticks(tick_dates)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%b'))
ax.minorticks_off()

ax.axvline(x=end_date_exact, linestyle=':', alpha=0.7, linewidth=1.5)

ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('OLS Hedge performance on historical backtest', fontsize=12, pad=20)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.subplots_adjust(bottom=0.22)
y_text = -0.12
ax.text(start_date,    y_text, '2022', transform=ax.get_xaxis_transform(), ha='left',   va='top', fontsize=10)
ax.text(year_boundary, y_text, '2023', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=10)

plt.tight_layout()
plt.setp(plt.gca().lines, linewidth=1.0)
plt.show()


# Save hedge
Hedge_out = pd.DataFrame(Hedge)
Hedge_out.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/OLS_test_hedge.xlsx',
                   index=False, header=False)

Rep_out = pd.DataFrame(proxy_values[:testing])
Rep_out.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/OLS_test_rep.xlsx',
                   index=False, header=False)
grads_out = pd.DataFrame(grads)
grads_out.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/OLS_deltas.xlsx',
                  index=False, header=False)

safe_weights_out = pd.DataFrame(safe_weights)
safe_weights_out.to_excel('/Users/cormactredoux/Documents/Uni/Proxy Hedging/Results Presentation/Raw/OLS_weights.xlsx',
                          index=False, header=False)    