import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#### config
FUND_NAME = "QQQ"
LOOKBACK = 25

RR_WANTED = True
EWLS_WANTED = True
EWMA_WANTED = True


def load_rr_results():
    folder_name = f"RR_Window_Size_Pretest_Lookback{LOOKBACK}"
    results_file = PROJECT_ROOT / "results" / FUND_NAME / folder_name / "window_size_pretest_results.xlsx"
    
    if not results_file.exists():
        print(f"Warning: RR file not found at {results_file}")
        return None
    return pd.read_excel(results_file)


def load_ewls_results():
    folder_name = f"EWLS_Beta_Pretest_Lookback{LOOKBACK}"
    results_file = PROJECT_ROOT / "results" / FUND_NAME / folder_name / "ewls_beta_pretest_results.xlsx"
    
    if not results_file.exists():
        print(f"Warning: EWLS file not found at {results_file}")
        return None
    return pd.read_excel(results_file)


def load_ewma_results():
    folder_name = f"EWMA_Alpha_Pretest_Lookback{LOOKBACK}"
    results_file = PROJECT_ROOT / "results" / FUND_NAME / folder_name / "ewma_alpha_pretest_results.xlsx"
    
    if not results_file.exists():
        print(f"Warning: EWMA file not found at {results_file}")
        return None
    return pd.read_excel(results_file)


def plot_combined_results():
    rr_df = load_rr_results() if RR_WANTED else None
    ewma_df = load_ewma_results() if EWMA_WANTED else None
    ewls_df = load_ewls_results() if EWLS_WANTED else None

    if rr_df is None and ewma_df is None and ewls_df is None:
        print("No data loaded. Exiting plot function.")
        return

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    row1_ylims, row2_ylims, row3_ylims, row4_ylims = [], [], [], []

    #### rolling regression
    if rr_df is not None:
        window_sizes = rr_df['window_size'].values
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[1, 0], axes[2, 0], axes[3, 0]
        
        best_idx = rr_df['abs_mean_tracking_error'].idxmin()
        best_window = rr_df.loc[best_idx, 'window_size']
        
        # row 1: abs mean TE
        ax1.plot(window_sizes, rr_df['abs_mean_tracking_error'], linewidth=2, color='red', marker='o', markersize=3)
        ax1.axvline(x=best_window, color='black', linestyle=':', linewidth=2, label=f'Optimal: {int(best_window)} days')
        ax1.set_xlabel('Window Size (days)'); ax1.set_ylabel('Abs Mean Tracking Error'); ax1.set_title('Rolling Regression', fontsize=24)
        ax1.grid(True, alpha=0.3); ax1.legend(fontsize=12); row1_ylims.extend(ax1.get_ylim())

        # row 2: sd TE
        ax2.plot(window_sizes, rr_df['sd_tracking_error'], linewidth=2, color='red', marker='o', markersize=3)
        ax2.axvline(x=best_window, color='black', linestyle=':', linewidth=2)
        ax2.set_xlabel('Window Size (days)'); ax2.set_ylabel('SD Tracking Error'); ax2.grid(True, alpha=0.3)
        row2_ylims.extend(ax2.get_ylim())
        
        # row 3: abs mean replication error
        ax3.plot(window_sizes, rr_df['abs_mean_replication_error'], linewidth=2, color='red', marker='o', markersize=3)
        ax3.axvline(x=best_window, color='black', linestyle=':', linewidth=2)
        ax3.set_xlabel('Window Size (days)'); ax3.set_ylabel('Abs Mean Replication Error'); ax3.grid(True, alpha=0.3)
        row3_ylims.extend(ax3.get_ylim())
        
        # row 4: sd replication error
        ax4.plot(window_sizes, rr_df['sd_replication_error'], linewidth=2, color='red', marker='o', markersize=3)
        ax4.axvline(x=best_window, color='black', linestyle=':', linewidth=2)
        ax4.set_xlabel('Window Size (days)'); ax4.set_ylabel('SD Replication Error'); ax4.grid(True, alpha=0.3)
        row4_ylims.extend(ax4.get_ylim())
    else:
        for ax in [axes[0,0], axes[1,0], axes[2,0], axes[3,0]]:
            ax.text(0.5,0.5,'RR Data Not Available', ha='center', va='center', fontsize=12)
            ax.axis('off')

    #### EWMA
    if ewma_df is not None:
        alphas = ewma_df['alpha'].values
        ax5, ax6, ax7, ax8 = axes[0,1], axes[1,1], axes[2,1], axes[3,1]
        
        best_idx = ewma_df['abs_mean_tracking_error'].idxmin()
        best_alpha = ewma_df.loc[best_idx,'alpha']
        
        # row 1: abs mean TE
        ax5.plot(alphas, ewma_df['abs_mean_tracking_error'], linewidth=2, color='green', marker='o', markersize=3)
        ax5.axvline(x=best_alpha, color='black', linestyle=':', linewidth=2, label=f'Optimal: {best_alpha}')
        ax5.set_xlabel('Alpha'); ax5.set_ylabel('Abs Mean Tracking Error'); ax5.set_title('EWMA (Window Size=40)', fontsize=24)
        ax5.grid(True, alpha=0.3); ax5.legend(fontsize=12); row1_ylims.extend(ax5.get_ylim())

        # row 2: sd TE
        ax6.plot(alphas, ewma_df['sd_tracking_error'], linewidth=2, color='green', marker='o', markersize=3)
        ax6.axvline(x=best_alpha, color='black', linestyle=':', linewidth=2)
        ax6.set_xlabel('Alpha'); ax6.set_ylabel('SD Tracking Error'); ax6.grid(True, alpha=0.3)
        row2_ylims.extend(ax6.get_ylim())
        
        # row 3: abs mean replication error
        ax7.plot(alphas, ewma_df['abs_mean_replication_error'], linewidth=2, color='green', marker='o', markersize=3)
        ax7.axvline(x=best_alpha, color='black', linestyle=':', linewidth=2)
        ax7.set_xlabel('Alpha'); ax7.set_ylabel('Abs Mean Replication Error'); ax7.grid(True, alpha=0.3)
        row3_ylims.extend(ax7.get_ylim())
        
        # row 4: sd replication error
        ax8.plot(alphas, ewma_df['sd_replication_error'], linewidth=2, color='green', marker='o', markersize=3)
        ax8.axvline(x=best_alpha, color='black', linestyle=':', linewidth=2)
        ax8.set_xlabel('Alpha'); ax8.set_ylabel('SD Replication Error'); ax8.grid(True, alpha=0.3)
        row4_ylims.extend(ax8.get_ylim())
    else:
        for ax in [axes[0,1], axes[1,1], axes[2,1], axes[3,1]]:
            ax.text(0.5,0.5,'EWMA Data Not Available', ha='center', va='center', fontsize=12)
            ax.axis('off')

    #### EWLS
    if ewls_df is not None:
        betas = ewls_df['beta'].values
        ax9, ax10, ax11, ax12 = axes[0,2], axes[1,2], axes[2,2], axes[3,2]
        
        best_idx = ewls_df['abs_mean_tracking_error'].idxmin()
        best_beta = ewls_df.loc[best_idx,'beta']
        
        # row 1: abs mean TE
        ax9.plot(betas, ewls_df['abs_mean_tracking_error'], linewidth=2, color='purple', marker='o', markersize=3)
        ax9.axvline(x=best_beta, color='black', linestyle=':', linewidth=2, label=f'Optimal: {best_beta}')
        ax9.set_xlabel('β (Beta)'); ax9.set_ylabel('Abs Mean Tracking Error'); ax9.set_title('EWLS (Window Size=575)', fontsize=24)
        ax9.grid(True, alpha=0.3); ax9.legend(fontsize=12); row1_ylims.extend(ax9.get_ylim())

        # row 2: sd TE
        ax10.plot(betas, ewls_df['sd_tracking_error'], linewidth=2, color='purple', marker='o', markersize=3)
        ax10.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
        ax10.set_xlabel('β (Beta)'); ax10.set_ylabel('SD Tracking Error'); ax10.grid(True, alpha=0.3)
        row2_ylims.extend(ax10.get_ylim())
        
        # row 3: abs mean replication error
        ax11.plot(betas, ewls_df['abs_mean_replication_error'], linewidth=2, color='purple', marker='o', markersize=3)
        ax11.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
        ax11.set_xlabel('β (Beta)'); ax11.set_ylabel('Abs Mean Replication Error'); ax11.grid(True, alpha=0.3)
        row3_ylims.extend(ax11.get_ylim())
        
        # row 4: sd replication error
        ax12.plot(betas, ewls_df['sd_replication_error'], linewidth=2, color='purple', marker='o', markersize=3)
        ax12.axvline(x=best_beta, color='black', linestyle=':', linewidth=2)
        ax12.set_xlabel('β (Beta)'); ax12.set_ylabel('SD Replication Error'); ax12.grid(True, alpha=0.3)
        row4_ylims.extend(ax12.get_ylim())
    else:
        for ax in [axes[0,2], axes[1,2], axes[2,2], axes[3,2]]:
            ax.text(0.5,0.5,'EWLS Data Not Available', ha='center', va='center', fontsize=12)
            ax.axis('off')

    #### set consistent y-limits per row
    if row1_ylims:
        y_min, y_max = min(row1_ylims), max(row1_ylims); y_pad = 0.05*(y_max-y_min)
        for ax in axes[0,:]: 
            if ax.lines: ax.set_ylim(y_min - y_pad, y_max + y_pad)
    if row2_ylims:
        y_min, y_max = min(row2_ylims), max(row2_ylims); y_pad = 0.05*(y_max-y_min)
        for ax in axes[1,:]:
            if ax.lines: ax.set_ylim(y_min - y_pad, y_max + y_pad)
    if row3_ylims:
        y_min, y_max = min(row3_ylims), max(row3_ylims); y_pad = 0.05*(y_max-y_min)
        for ax in axes[2,:]:
            if ax.lines: ax.set_ylim(y_min - y_pad, y_max + y_pad)
    if row4_ylims:
        y_min, y_max = min(row4_ylims), max(row4_ylims); y_pad = 0.05*(y_max-y_min)
        for ax in axes[3,:]:
            if ax.lines: ax.set_ylim(y_min - y_pad, y_max + y_pad)

    plt.tight_layout()
    
    output_dir = PROJECT_ROOT / "final_plots" / "hyperparam"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{FUND_NAME}_hyperparam_tuning_lookback{LOOKBACK}.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_combined_results()