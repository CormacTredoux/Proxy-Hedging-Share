import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from RR import RollingRegressionModel 
from utils import read_xlsx_file, format_time_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def proxy_folder_name(proxy_cols):
    #### automated saving conventions; sorted names
    return "-".join(sorted(proxy_cols))


def run_simulation():
    #### config
    data_file = 'Combined_data_new.xlsx'
    fund_name = "QQQ"
    proxy_cols = ["MXEU", "MXEF", "MXEA"]
    window_size = 600
    start_day = 600
    testing_size = 252
    r = 0.05

    #### output directory
    proxy_folder = proxy_folder_name(proxy_cols)
    output_dir = os.path.join(PROJECT_ROOT, "results", fund_name, proxy_folder, "Rolling", "Test Set")
    os.makedirs(output_dir, exist_ok=True)

    #### load data and compute strike
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
        raise IndexError(f"Data length ({len(all_dates)}) insufficient for start_day ({start_day}) + testing_size ({testing_size}).")
    
    dates = all_dates[start_day : start_day + testing_size]

    #### run model
    model = RollingRegressionModel(
        data_file=data_file,
        fund_name=fund_name,
        proxy_cols=proxy_cols,
        window_size=window_size,
        start_day=start_day,
        testing_size=testing_size,
        r=r
    )
    
    results = model.run_hedge(Strike)

    #### compute tracking errors
    manual_tracking_errors = np.zeros(testing_size)
    for t in range(1, testing_size):
        hedge_change = results['hedge_values'][t] - results['hedge_values'][t-1]
        liability_change = results['liability_values'][t] - results['liability_values'][t-1]
        manual_tracking_errors[t] = hedge_change - liability_change

    #### combine results
    combined_df = pd.DataFrame({
        'Date': dates,
        'Fund_Actual': results['fund_actual'],
        'Replication': results['replication'],
        'Liability_Value': results['liability_values'],
        'Hedge_Value': results['hedge_values'],
        'Tracking_Error': manual_tracking_errors,
        'Gradient': results['gradients'],
    })

    for i, col in enumerate(proxy_cols):
        combined_df[f'Hedge_Ratio_{col}'] = results['weights'][:, i]
        combined_df[f'Hedge_Weight_{col}'] = results['gradients'] * results['weights'][:, i]

    #### save results
    combined_file = os.path.join(output_dir, f'rolling_window_combined-w{window_size}.xlsx')
    combined_df.to_excel(combined_file, index=False, engine='openpyxl')

    #### plots

    # replication
    fig0, ax0 = plt.subplots(figsize=(12, 6))
    ax0.plot(dates, results['fund_actual'], label='Actual Fund', color='black', linewidth=2)
    ax0.plot(dates, results['replication'], label='Replication Portfolio', color='red', linewidth=2)
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Value')
    ax0.set_title('Rolling Regression - Proxy Quality')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    format_time_axis(ax0, dates, tick_count=7, year_labels=True)
    plt.tight_layout()

    # hedge vs liability
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(dates, results['hedge_values'], label='Proxy Hedge Value', color='blue', linewidth=1.5)
    ax1.plot(dates, results['liability_values'], label='Proxy Model Liability', color='green', linewidth=1.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.set_title('Rolling Regression - Hedge vs Liability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    format_time_axis(ax1, dates, tick_count=7, year_labels=True)
    plt.tight_layout()

    # tracking error
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(dates, manual_tracking_errors, label='Tracking Error',
             color='purple', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Tracking Error')
    ax3.set_title('Rolling Regression - Tracking Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    format_time_axis(ax3, dates, tick_count=7, year_labels=True)
    plt.tight_layout()

    # weight progression
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'orange', 'green']
    for i, col in enumerate(proxy_cols):
        ax4.plot(dates, results['weights'][:, i], label=f'{col} Weight', 
                color=colors[i % len(colors)], linewidth=2)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Weight (Hedge Ratio)')
    ax4.set_title('Rolling Regression - Weight Progression')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    format_time_axis(ax4, dates, tick_count=7, year_labels=True)
    plt.tight_layout()

    plt.show()
    return results


if __name__ == '__main__':
    run_simulation()