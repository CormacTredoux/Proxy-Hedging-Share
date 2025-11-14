import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pathlib import Path
from utils import read_xlsx_file
from Simulations import sim_year


#### config
data_file = 'Combined_data_new.xlsx'
fund_name = 'QQQ'
proxy_cols = ['CSCO', 'QCOM']
test_start = 600
test_len = 252
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
save_dir = Path(PROJECT_ROOT) / 'final_plots' / 'Cholesky'
save_dir.mkdir(parents=True, exist_ok=True)
dt = 1 / 252
r = 0.05
seed = 123
np.random.seed(seed)

asset_colors = ['cyan', 'magenta']

#### load data
data = read_xlsx_file(data_file)
proxies = data[proxy_cols].to_numpy()
fund = data[fund_name].to_numpy()

#### extract trading dates
try:
    all_dates = pd.to_datetime(data.iloc[:, 0].values)
except Exception as e:
    all_dates = pd.to_datetime(pd.RangeIndex(start=0, stop=len(data), step=1), unit='D')

num_hist_days = 100
if len(all_dates) < test_start + test_len:
    raise IndexError(f"Data length ({len(all_dates)}) insufficient")
if test_start < num_hist_days:
    raise IndexError(f"test_start ({test_start}) too small")

trading_ix = all_dates[test_start : test_start + test_len]
hist_ix = all_dates[test_start - num_hist_days : test_start]

trading_ix = pd.DatetimeIndex(trading_ix)
hist_ix = pd.DatetimeIndex(hist_ix)

#### volatility calibration
calib_window = 100
proxies_subset = proxies[test_start - calib_window:test_start]
n = len(proxy_cols)
T_sim = test_len

delta = np.diff(proxies_subset, axis=0)
sigma_annual = delta.std(ddof=1, axis=0) * np.sqrt(252)

#### independent simulations
S_indep = np.zeros((T_sim + 1, n))
S_indep[0] = proxies_subset[-1]
Z_raw = np.random.randn(T_sim, n)

for t in range(T_sim):
    dW = Z_raw[t] * np.sqrt(dt)
    S_indep[t + 1] = S_indep[t] + r * S_indep[t] * dt + sigma_annual * dW

#### correlated simulations
rho = 0.8
corr_matrix = np.full((n, n), rho)
np.fill_diagonal(corr_matrix, 1.0)
L_corr = np.linalg.cholesky(corr_matrix)

dS = np.diff(S_indep, axis=0)
dS_corr = dS @ L_corr.T
S_corr = np.zeros_like(S_indep)
S_corr[0] = S_indep[0]
for t in range(T_sim):
    S_corr[t + 1] = S_corr[t] + dS_corr[t]

#### helper function for date formatting
def format_ax(ax, dates, title=""):
    ax.set_title(title)
    months = pd.date_range(start=dates[0], end=dates[-1], freq='MS')[::2]
    ax.set_xticks(months)
    ax.xaxis.set_major_formatter(DateFormatter('%d/%b'))
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylabel('Price', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=12)

    ylim = ax.get_ylim()
    for year_start in pd.date_range(start=dates[0], end=dates[-1], freq='YS'):
        if year_start >= dates[0] and year_start <= dates[-1]:
            ax.text(year_start, ylim[0] - 0.06*(ylim[1]-ylim[0]), str(year_start.year),
                    ha='center', va='top', fontsize=10)

#### plot 1 & 2: independent and correlated
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# left: independent
for i, col in enumerate(proxy_cols):
    ax1.plot(trading_ix, S_indep[1:, i], lw=2, label=col, color=asset_colors[i])
format_ax(ax1, trading_ix, "Uncorrelated Simulation")

# right: correlated
for i, col in enumerate(proxy_cols):
    ax2.plot(trading_ix, S_corr[1:, i], lw=2, label=col, color=asset_colors[i])
format_ax(ax2, trading_ix, "Correlated Simulation, Cov = 0.8")

plt.subplots_adjust(bottom=0.2)
plot12_path = save_dir / "plot12_proxy_comparison.png"
plt.savefig(plot12_path, dpi=300, bbox_inches='tight')
plt.show()

#### plot 3: QQQ history + future simulations
fund_hist = fund[test_start - 100:test_start]
fund_test = fund[test_start:test_start + test_len + 1]
T_sim = 252

F_sims = []
for i in range(1000):
    _, F_sim = sim_year(proxies[test_start - 100:test_start], fund[test_start - 100:test_start], r, T_sim)
    F_sims.append(F_sim)
F_sims = np.array(F_sims)

fig, ax = plt.subplots(figsize=(10, 6))
dates_future = trading_ix
dates_hist   = hist_ix
dates_full   = dates_hist.append(dates_future)

fund_full = np.concatenate([fund_hist, fund_test[1:]])

for i in range(300):
    ax.plot(dates_future, F_sims[i, 1:], color='grey', alpha=0.25, lw=1.1)
ax.plot(dates_full, fund_full, color='black', lw=2.8, label='QQQ (calibration + realisation)')

format_ax(ax, dates_full, "Historical + Future Fund Sims")
plt.subplots_adjust(bottom=0.2)
plot3_path = save_dir / "plot3_QQQ_sims.png"
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.show()