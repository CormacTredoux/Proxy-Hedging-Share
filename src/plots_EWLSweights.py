import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


#### config
window_size = 600
beta_1 = 0.9997
beta_2 = 0.997


def compute_weights(window_size, beta):
    #### compute and normalise EWLS weights
    weights = beta ** np.arange(window_size - 1, -1, -1)
    weights_normalised = weights / weights.sum()
    return weights_normalised


def main():
    observations = np.arange(1, window_size + 1)
    
    weights_1 = compute_weights(window_size, beta_1)
    weights_2 = compute_weights(window_size, beta_2)
    
    ess_1 = 1 / (weights_1**2).sum()
    ess_2 = 1 / (weights_2**2).sum()
    
  
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # top: normalised weights with beta = 0.9997
    ax1.bar(observations, weights_1, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Observation Number (1 = Oldest, {} = Most Recent)'.format(window_size), fontsize=12)
    ax1.set_ylabel('Normalised Weight', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # bottom: normalised weights with beta = 0.997
    ax2.bar(observations, weights_2, color='darkorange', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Observation Number (1 = Oldest, {} = Most Recent)'.format(window_size), fontsize=12)
    ax2.set_ylabel('Normalised Weight', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    #### save plot
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = Path(PROJECT_ROOT) / 'final_plots' / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_file = save_dir / f"ewls_weights_comparison_w{window_size}.png"
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    main()