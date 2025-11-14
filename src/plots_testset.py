import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import format_time_axis, finalize_plot, read_xlsx_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_model_xlsx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_excel(file_path)


def proxy_folder_name(proxy_cols):
    return "-".join(sorted(proxy_cols))


#### configuration
fund_name   = "QQQ"
proxy_cols  = ["M1US000G", "M6US0IN", "MXUS0IT"]
testing_size = 252
training     = 400
test_start   = 600

proxy_folder = proxy_folder_name(proxy_cols)

#### paths
base_path = PROJECT_ROOT / "results" / fund_name / proxy_folder
kalman_file  = base_path / "Kalman" / "Test Set" / "kalman_filter_combined.xlsx"
ols_file     = base_path / "OLS" / "Test Set" / "OLS_combined-w400.xlsx"
rolling_file = base_path / "Rolling" / "Test Set" / "rolling_window_combined-w600.xlsx"

#### load model test-set data
Kalman_df = read_model_xlsx(kalman_file)
RW_df     = read_model_xlsx(rolling_file)
OLS_df    = read_model_xlsx(ols_file)

Kalman_hedge = Kalman_df["Hedge_Value"].to_numpy()
RW_hedge     = RW_df["Hedge_Value"].to_numpy()
OLS_hedge    = OLS_df["Hedge_Value"].to_numpy()
Kalman_liability = Kalman_df["Liability_Value"].to_numpy()
RW_liability     = RW_df["Liability_Value"].to_numpy()

if not np.allclose(Kalman_liability, RW_liability, rtol=0, atol=1e-5):
    raise ValueError("Liabilities from Kalman and Rolling Regression do not match")

Liability = Kalman_liability

Kalman_rep = Kalman_df["Replication"].to_numpy()
RW_rep     = RW_df["Replication"].to_numpy()
OLS_rep    = OLS_df["Replication"].to_numpy()
Fund_actual = Kalman_df["Fund_Actual"].to_numpy()

#### extract corresponding dates
financial_data_file = PROJECT_ROOT / "Clean_Dat" / "Combined_data_new.xlsx"
data = read_xlsx_file(os.path.basename(financial_data_file))

try:
    date_series = pd.to_datetime(data.iloc[:, 0].iloc[test_start:test_start + testing_size])
except Exception as e:
    date_series = pd.to_datetime(pd.RangeIndex(start=test_start, stop=test_start+testing_size, step=1), unit='D')

date_range = np.array(date_series.dt.to_pydatetime())

#### align arrays
n0 = min(len(date_range), len(Liability), len(RW_hedge), len(OLS_hedge), len(Kalman_hedge),
         len(Fund_actual), len(Kalman_rep), len(RW_rep), len(OLS_rep))

date_plot = date_range[:n0]
V_plot    = Liability[:n0]
RW_plot   = RW_hedge[:n0]
OLS_plot  = OLS_hedge[:n0]
Kal_plot  = Kalman_hedge[:n0]
Fund_plot = Fund_actual[:n0]
RW_rep_plot = RW_rep[:n0]
OLS_rep_plot = OLS_rep[:n0]
Kal_rep_plot = Kalman_rep[:n0]

#### create 1x2 figure
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# left: replication quality
axs[0].plot(date_plot, Fund_plot,    label='Fund Value', color='black', linewidth=2)
axs[0].plot(date_plot, RW_rep_plot,  label='RWOLS',        color='red', linewidth=1.2)
axs[0].plot(date_plot, OLS_rep_plot, label='OLS',          color='blue', linewidth=1.2)
axs[0].plot(date_plot, Kal_rep_plot, label='Kalman',       color='orange', linewidth=1.2)

format_time_axis(axs[0], date_plot, tick_count=5, year_labels=True)
finalize_plot(axs[0], title='Proxy Quality Comparison', xlabel='Date')

# right: hedge vs liability
axs[1].plot(date_plot, V_plot,   label='Liability', color='black', linewidth=2)
axs[1].plot(date_plot, RW_plot,  label='RWOLS',     color='red', linewidth=1.2)
axs[1].plot(date_plot, OLS_plot, label='OLS',       color='blue', linewidth=1.2)
axs[1].plot(date_plot, Kal_plot, label='Kalman',    color='orange', linewidth=1.2)

format_time_axis(axs[1], date_plot, tick_count=5, year_labels=True)
finalize_plot(axs[1], title='Hedge vs Liability Comparison', xlabel='Date')

for ax in axs:
    ax.axvline(x=date_plot[-1], color='black', linestyle=':', alpha=0.7, linewidth=1.2)

#### save plot
final_plot_dir = PROJECT_ROOT / "final_plots" / "Asset Selection"
os.makedirs(final_plot_dir, exist_ok=True)
proxy_str = "-".join(sorted(proxy_cols))
file_name = f"comparison_{fund_name}_{proxy_str}.png"
save_path = final_plot_dir / file_name
fig.tight_layout()
fig.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()