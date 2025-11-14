import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PCA_RR_dynamic import RollingRegressionPCAModel
from Hedging import Hedge_F_self
from Pricing import Price_F
from utils import read_xlsx_file, format_time_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]

#### color palette for weight progression
COLORS = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'maroon', 'teal']


def run_optimal_fund_hedge(model_instance, Strike, sigma_fund):
    #### calculate optimal self-financing hedge using underlying fund
    fund_test = model_instance.fund[model_instance.start_day:model_instance.start_day + model_instance.testing_size]
    T = model_instance.testing_size
    r = model_instance.r

    tracking_errors_fund, Hedge_Fund, V_Fund, grads_fund = Hedge_F_self(
        Strike=Strike,
        fund=fund_test,
        sigma=sigma_fund,
        r=r,
        T=T,
        Price_F=Price_F 
    )
    
    return Hedge_Fund, tracking_errors_fund


def run_simulation():
    #### configuration
    data_file = 'Combined_data_new.xlsx'
    fund_name = "BBF"
    window_size = 210
    start_day = 600
    testing_size = 252
    r = 0.05

    df = read_xlsx_file(data_file)
    fund_data = df[fund_name].values
    
    F_0 = fund_data[start_day - 1]
    T_years = testing_size / 252
    Strike = F_0 * np.exp(r * T_years)

    #### extract dates
    try:
        all_dates = pd.to_datetime(df.iloc[:, 0].values)
    except Exception:
        all_dates = pd.to_datetime(pd.RangeIndex(start=0, stop=len(df), step=1), unit='D')

    if len(all_dates) < start_day + testing_size:
        raise IndexError(f"Data length ({len(all_dates)}) insufficient for start_day ({start_day}) + testing_size ({testing_size})")
    
    dates = all_dates[start_day : start_day + testing_size]

    #### run PCA-based dynamic rolling regression
    model = RollingRegressionPCAModel(
        data_file=data_file,
        fund_name=fund_name,
        window_size=window_size,
        start_day=start_day,
        testing_size=testing_size,
        r=r
    )

    try:
        _, _, sigma_fund = model.compute_volatility_params()
    except Exception:
        calib_start = start_day - 400
        calib_end = start_day
        fund_train = df[fund_name].values[calib_start:calib_end]
        delta_y = np.diff(fund_train)
        sigma_fund = delta_y.std(ddof=1) * np.sqrt(252)

    results = model.run_hedge(Strike)
    
    Hedge_Fund, tracking_errors_fund = run_optimal_fund_hedge(model, Strike, sigma_fund)

    #### compute tracking errors
    manual_tracking_errors = np.zeros(testing_size)
    for t in range(1, testing_size):
        hedge_change = results['hedge_values'][t] - results['hedge_values'][t-1]
        liability_change = results['liability_values'][t] - results['liability_values'][t-1]
        manual_tracking_errors[t] = hedge_change - liability_change

    try:
        asset_cols = model.all_proxy_cols
    except AttributeError:
        asset_cols = model.all_assets 
    
    #### combine results
    combined_df = pd.DataFrame({
        'Date': dates,
        'Fund_Actual': results['fund_actual'],
        'Replication': results['replication'],
        'Liability_Value': results['liability_values'],
        'Hedge_Value': results['hedge_values'],
        'Tracking_Error': manual_tracking_errors,
        'Gradient': results['gradients'],
        'Fund_Hedge_Value': Hedge_Fund,
        'Fund_Tracking_Error': tracking_errors_fund,
    })

    for i, col in enumerate(asset_cols):
        combined_df[f'Hedge_Ratio_{col}'] = results['weights'][:, i]
        combined_df[f'Hedge_Weight_{col}'] = results['gradients'] * results['weights'][:, i]

    #### save results
    output_dir = os.path.join(PROJECT_ROOT, "results", fund_name, "PCA_Rolling", "Test Set")
    os.makedirs(output_dir, exist_ok=True)
    combined_file = os.path.join(output_dir, f'rolling_window_PCA_combined-w{window_size}.xlsx')
    combined_df.to_excel(combined_file, index=False, engine='openpyxl')

    #### 2x2 plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # top left: replication vs fund
    ax0 = axs[0, 0]
    ax0.plot(dates, results['fund_actual'], label=f'{fund_name} Fund Value', color='black', linewidth=2)
    ax0.plot(dates, results['replication'], label='RR-PCA Replication', color='red', linewidth=2) 
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Value')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    format_time_axis(ax0, dates, tick_count=5, year_labels=False)

    # top right: hedge vs liability
    ax1 = axs[0, 1]
    ax1.plot(dates, results['liability_values'], label='Liability Value', color='black', linewidth=2)
    ax1.plot(dates, results['hedge_values'], label='RR-PCA Hedge Value', color='red', linewidth=2) 
    ax1.plot(dates, Hedge_Fund, label='Fund Hedge (Optimal)', color='blue', linewidth=2, linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    format_time_axis(ax1, dates, tick_count=5, year_labels=False)

    # bottom left: weight progression
    ax2 = axs[1, 0]
    for i, col in enumerate(asset_cols):
        if np.any(results['weights'][:, i] != 0):
             ax2.plot(dates, results['weights'][:, i], 
                      label=f'{col} Weight', 
                      color=COLORS[i % len(COLORS)], 
                      linewidth=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Weight (Hedge Ratio)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    format_time_axis(ax2, dates, tick_count=5, year_labels=True)

    # bottom right: tracking error
    ax3 = axs[1, 1]
    ax3.plot(dates, manual_tracking_errors, label='RR-PCA Tracking Error', color='red', linewidth=2)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Tracking Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    format_time_axis(ax3, dates, tick_count=5, year_labels=True)

    plt.tight_layout() 
    
    plt.show()

    return combined_df


if __name__ == "__main__":
    results_df = run_simulation()