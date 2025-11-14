import numpy as np
import pandas as pd
from pathlib import Path
from EWLS import EWLSModel
from utils import read_xlsx_file
import matplotlib.pyplot as plt
from tqdm import tqdm


#### 0 = pre-test validation, 1 = post-hoc analysis, 2 = run both
POSTHOC = 0
LOOKBACK = 25


def run_beta_analysis(mode='posthoc', lookback=LOOKBACK):
    data_file = 'Combined_data_new.xlsx'
    fund_name = "QQQ"
    proxy_cols = ["CSCO", "RFL", "QCOM"]
    start_day = 600
    testing_size = 252
    r = 0.05
    window_size = 600
    
    #### determine parameters based on mode
    if mode == 'pretest':
        max_window = start_day - lookback
        if window_size > max_window:
            actual_window = max_window
        else:
            actual_window = window_size
        
        test_start = start_day - lookback
        test_size = lookback
        analysis_name = f"Pre-test Validation (Lookback={lookback})"
        output_subdir = f"EWLS_Beta_Pretest_Lookback{lookback}"
    else:
        actual_window = window_size
        test_start = start_day
        test_size = testing_size
        analysis_name = "Post-hoc Analysis"
        output_subdir = "EWLS_Beta_Analysis"
    
    betas = np.arange(0.970, 1, 0.0001)
    betas = np.round(betas, 4)
    betas = betas[betas <= 1.0]
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    output_dir = PROJECT_ROOT / "results" / fund_name / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #### load data and compute strike
    df = read_xlsx_file(data_file)
    fund_data = df[fund_name].values
    F_0 = fund_data[test_start - 1]
    T_years = test_size / 252
    Strike = F_0 * np.exp(r * T_years)
    
    results_list = []
    
    #### run simulation for each beta value
    for beta in tqdm(betas, desc=f"Processing beta values ({mode})"):
        try:
            model = EWLSModel(
                data_file=data_file,
                fund_name=fund_name,
                proxy_cols=proxy_cols,
                window_size=actual_window,
                decay=beta,
                start_day=test_start,
                testing_size=test_size,
                r=r
            )
            results = model.run_hedge(Strike)
            
            hedge_values = results['hedge_values']
            liability_values = results['liability_values']
            
            manual_tracking_errors = np.zeros(test_size)
            for t in range(1, test_size):
                hedge_change = hedge_values[t] - hedge_values[t-1]
                liability_change = liability_values[t] - liability_values[t-1]
                manual_tracking_errors[t] = hedge_change - liability_change
            
            replication_errors = results['fund_actual'] - results['replication']
            
            results_list.append({
                'beta': beta,
                'window_size': actual_window,
                'mean_tracking_error': np.mean(manual_tracking_errors),
                'abs_mean_tracking_error': np.mean(np.abs(manual_tracking_errors)),
                'mean_replication_error': np.mean(replication_errors),
                'abs_mean_replication_error': np.mean(np.abs(replication_errors)),
                'sd_tracking_error': np.std(manual_tracking_errors),
                'sd_replication_error': np.std(replication_errors)
            })
            
        except Exception:
            continue
    
    results_df = pd.DataFrame(results_list)
    
    #### save results
    results_file = output_dir / f'ewls_beta_{mode}_results.xlsx'
    results_df.to_excel(results_file, index=False)
    
    create_analysis_plots(results_df, output_dir, analysis_name, mode, actual_window)
    
    best_abs_te = results_df.loc[results_df['abs_mean_tracking_error'].idxmin()]
    
    return results_df, best_abs_te['beta']


def create_analysis_plots(results_df, output_dir, analysis_name, mode, window_size):
    betas = results_df['beta'].values
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f'EWLS Beta Parameter Analysis (Window Size={window_size})\n{analysis_name}', fontsize=14, y=0.995)
    
    best_idx = results_df['abs_mean_tracking_error'].idxmin()
    best_beta = results_df.loc[best_idx, 'beta']
    
    # top left: mean tracking error
    ax1 = axes[0, 0]
    ax1.plot(betas, results_df['mean_tracking_error'], linewidth=2, color='blue', marker='o', markersize=3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
    ax1.set_ylabel('Mean Tracking Error', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # top right: absolute mean tracking error
    ax2 = axes[0, 1]
    ax2.plot(betas, results_df['abs_mean_tracking_error'], linewidth=2, color='red', marker='o', markersize=3)
    ax2.axvline(x=best_beta, color='black', linestyle=':', linewidth=2, label=f"Optimal: {best_beta:.4f}")
    ax2.set_ylabel('Absolute Mean Tracking Error', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9) 
    
    # middle left: sd tracking error
    ax3 = axes[1, 0]
    ax3.plot(betas, results_df['sd_tracking_error'], linewidth=2, color='orange', marker='o', markersize=3)
    ax3.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
    ax3.set_ylabel('SD Tracking Error', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # middle right: mean replication error
    ax4 = axes[1, 1]
    ax4.plot(betas, results_df['mean_replication_error'], linewidth=2, color='green', marker='o', markersize=3)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
    ax4.set_ylabel('Mean Replication Error', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    ## bottom left: absolute mean replication error
    ax5 = axes[2, 0]
    ax5.plot(betas, results_df['abs_mean_replication_error'], linewidth=2, color='purple', marker='o', markersize=3)
    ax5.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
    ax5.set_xlabel('Beta Parameter', fontsize=11)
    ax5.set_ylabel('Absolute Mean Replication Error', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # bottom right: sd replication error
    ax6 = axes[2, 1]
    ax6.plot(betas, results_df['sd_replication_error'], linewidth=2, color='brown', marker='o', markersize=3)
    ax6.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
    ax6.set_xlabel('Beta Parameter', fontsize=11)
    ax6.set_ylabel('SD Replication Error', fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / f'ewls_beta_{mode}_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    plt.show(block=False)
    plt.pause(0.1)


if __name__ == '__main__':
    if POSTHOC == 0:
        results_df, best_beta = run_beta_analysis(mode='pretest', lookback=LOOKBACK)
        plt.show()
        
    elif POSTHOC == 1:
        results_df, _ = run_beta_analysis(mode='posthoc', lookback=LOOKBACK)
        plt.show()
        
    elif POSTHOC == 2:
        pretest_df, best_beta = run_beta_analysis(mode='pretest', lookback=LOOKBACK)
        posthoc_df, _ = run_beta_analysis(mode='posthoc', lookback=LOOKBACK)
        plt.show()