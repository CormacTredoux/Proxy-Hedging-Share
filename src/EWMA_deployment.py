import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from EWMA import EWMAModel
from utils import read_xlsx_file, format_time_axis
import os


def run_simulation():
    #### configuration
    data_file = 'Combined_data_new.xlsx'
    fund_name = "QQQ"
    proxy_cols = ["CSCO", "RFL", "QCOM"]
    alpha = 0.064
    window_size = 40
    start_day = 600
    testing_size = 252
    r = 0.05

    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = Path(script_dir) / "results" / fund_name / "Test Set" / "EWMA"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        raise IndexError(f"Data length insufficient")
    
    dates = all_dates[start_day : start_day + testing_size]

    #### run model with selected alpha
    model = EWMAModel(
        data_file=data_file,
        fund_name=fund_name,
        proxy_cols=proxy_cols,
        alpha=alpha,
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
    combined_df = pd.DataFrame({'Date': dates})
    combined_df['Fund_Actual'] = results['fund_actual']
    combined_df['Replication'] = results['replication']
    combined_df['Liability_Value'] = results['liability_values']
    combined_df['Hedge_Value'] = results['hedge_values']
    combined_df['Tracking_Error'] = manual_tracking_errors
    combined_df['Gradient'] = results['gradients']

    for i, col in enumerate(proxy_cols):
        combined_df[f'Hedge_Ratio_{col}'] = results['weights'][:, i]
        combined_df[f'Hedge_Weight_{col}'] = results['gradients'] * results['weights'][:, i]

    combined_file = output_dir / f'ewma_combined_w{window_size}_alpha{alpha}.xlsx'
    combined_df.to_excel(combined_file, index=False)

    #### plots
    
    # replication quality
    fig0, ax0 = plt.subplots(figsize=(12, 6))
    ax0.plot(dates, results['fund_actual'], label='Actual Fund', color='blue', linewidth=2)
    ax0.plot(dates, results['replication'], label='Replication Portfolio', color='orange', linewidth=2)
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Value')
    ax0.set_title('EWMA - Proxy Quality')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    format_time_axis(ax0, dates, tick_count=7, year_labels=True)
    plt.tight_layout()

    # hedge vs liability
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(dates, results['hedge_values'], label='Hedge Portfolio Value', color='blue', linewidth=2)
    ax1.plot(dates, results['liability_values'], label='Liability Value', color='green', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.set_title('EWMA - Hedge Performance')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    format_time_axis(ax1, dates, tick_count=7, year_labels=True)
    plt.tight_layout()

    # tracking error
    fig_te, ax_te = plt.subplots(figsize=(12, 5))
    ax_te.plot(dates, manual_tracking_errors, label='Tracking Error', color='green', linewidth=1.5)
    ax_te.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_te.set_xlabel('Date')
    ax_te.set_ylabel('Tracking Error')
    ax_te.set_title('EWMA - Tracking Error')
    ax_te.legend()
    ax_te.grid(True, alpha=0.3)
    format_time_axis(ax_te, dates, tick_count=7, year_labels=True)
    plt.tight_layout()

    save_dir_plot = Path(script_dir) / "final_plots" / "hyperparam"
    save_dir_plot.mkdir(parents=True, exist_ok=True)
    te_plot_path = save_dir_plot / "EWMA_TE_w40.png"
    fig_te.savefig(te_plot_path, dpi=300, bbox_inches='tight')

    #### alpha = 1 model for comparison
    model_full_alpha = EWMAModel(
        data_file=data_file,
        fund_name=fund_name,
        proxy_cols=proxy_cols,
        alpha=1.0,
        window_size=window_size,
        start_day=start_day,
        testing_size=testing_size,
        r=r
    )
    results_full = model_full_alpha.run_hedge(Strike)

    #### weight progression comparison (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = ['blue', 'orange', 'green']

    # left: alpha = 1
    for i, col in enumerate(proxy_cols):
        ax1.plot(dates, results_full['weights'][:, i], label=col, color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Weight')
    ax1.set_title(r'$\alpha = 1$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    format_time_axis(ax1, dates, tick_count=7, year_labels=True)

    # right: alpha = selected
    for i, col in enumerate(proxy_cols):
        ax2.plot(dates, results['weights'][:, i], label=col, color=colors[i % len(colors)], linewidth=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Weight')
    ax2.set_title(fr'$\alpha = {alpha}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    format_time_axis(ax2, dates, tick_count=7, year_labels=True)

    plt.tight_layout()

    trajectory_path = save_dir_plot / "EWMA_trajectory.png"
    fig.savefig(trajectory_path, dpi=300, bbox_inches='tight')

    plt.show()
    return results


if __name__ == '__main__':
    run_simulation()