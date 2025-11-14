import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from utils import read_xlsx_file
from Simulations import sim_year
from multiprocessing import Pool
from RR import RollingRegressionModel
from Hedging import Hedge_F
from Pricing import Price_F


#### global vars
Proxies = None
Fund = None
Proxies_train = None
Fund_train = None
proxy_cols = None
window_size = None
testing = None
start_test = None
start_train = None
r = None
file_path = None
fund_name = None


def init_thread(proxies, fund, proxies_train, fund_train, cols, w_size, test, s_test, s_train, rate, fpath, fname):
    #### initialise threads, luckily not much finesse needed here since each thread works independantly 
    global Proxies, Fund, Proxies_train, Fund_train, proxy_cols, window_size, testing, start_test, start_train, r, file_path, fund_name
    Proxies = proxies
    Fund = fund
    Proxies_train = proxies_train
    Fund_train = fund_train
    proxy_cols = cols
    window_size = w_size
    testing = test
    start_test = s_test
    start_train = s_train
    r = rate
    file_path = fpath
    fund_name = fname


def run_single_simulation(sim_idx):
    T_sim = 252

    #### simulate test year from bootstrap training slice
    X, Y = sim_year(Proxies_train, Fund_train, r, T_sim)
    
    #### combine training and simulated test data
    X_combined = np.vstack([Proxies[start_train:start_test, :], X.T])
    Y_combined = np.concatenate([Fund[start_train:start_test], Y])
    
    total_length = len(Y_combined)
    required_length = window_size + testing
    
    if total_length < required_length:
        raise ValueError(f"Insufficient data: have {total_length}, need {required_length}")
    
    temp_data = pd.DataFrame(X_combined, columns=proxy_cols)
    temp_data[fund_name] = Y_combined
    
    #### create model instance with simulated data
    model = RollingRegressionModel.__new__(RollingRegressionModel)
    model.fund_name = fund_name
    model.proxy_cols = proxy_cols
    model.window_size = window_size
    model.start_day = window_size
    model.testing_size = testing
    model.r = r
    model.data = temp_data
    model.assets = X_combined
    model.fund = Y_combined
    
    #### compute strike
    year = 252
    end_training_idx = window_size - 1
    tau = testing / year
    Strike = float(Y_combined[end_training_idx]) * np.exp(r * tau)
    
    #### compute volatility parameters from actual training window
    train_end = window_size
    train_start_local = 0
    
    assets_train = X_combined[train_start_local:train_end, :]
    fund_train = Y_combined[train_start_local:train_end]
    
    delta = np.diff(assets_train, axis=0)
    sigma_annual = delta.std(axis=0, ddof=1) * np.sqrt(252)
    corr = np.corrcoef(delta, rowvar=False)
    
    delta_y = np.diff(fund_train)
    sigma_fund = delta_y.std() * np.sqrt(252)
    
    assets_test = X_combined[window_size:window_size + testing, :].T
    fund_test = Y_combined[window_size:window_size + testing]
    
    rolling_weights = model.compute_rolling_weights()
    
    #### run hedging
    from Hedging import ProxyHedge_self
    from Pricing import Price
    
    tracking_errors, hedge_values, Vp, grads = ProxyHedge_self(
        Strike=Strike,
        assets=assets_test,
        weights=rolling_weights,
        sigma_annual=sigma_annual,
        corr=corr,
        r=r,
        T=testing,
        Price=Price,
        sigma_fund=sigma_fund,
        fund=fund_test
    )
    
    tracking_errors2, _, liability_values, _ = Hedge_F(
        Strike=Strike,
        fund=fund_test,
        sigma=sigma_fund,
        r=r,
        T=testing,
        Price_F=Price_F
    )
    
    #### compute tracking error metrics
    final_te = hedge_values[testing - 1] - hedge_values[testing - 2] - \
               (liability_values[testing - 1] - liability_values[testing - 2])
    avg_te = np.mean(np.diff(hedge_values) - np.diff(liability_values))
    
    return final_te, avg_te


if __name__ == '__main__':
    np.random.seed(941)
    random.seed(941)
    
    #### config
    file = "Combined_data_new.xlsx"
    fund_name = "QQQ"
    proxy_cols_main = ["CSCO", "RFL", "QCOM"]
    num_proxies = len(proxy_cols_main)
    data = read_xlsx_file(file)

    bach_train = 100
    Proxies_main = data[proxy_cols_main].to_numpy()
    Proxies_train_main = Proxies_main[300:300 + bach_train, :]
    Fund_main = data[fund_name].to_numpy()
    Fund_train_main = Fund_main[300:300 + bach_train]

    r_main = 0.05

    window_size_main = 500
    testing_main = 252
    start_test_main = 600
    start_train_main = start_test_main - window_size_main

    n_sim = 10000
    n_threads = 10

    if start_train_main < 0:
        raise ValueError(f"start_train ({start_train_main}) is negative")
    if start_test_main > len(Fund_main):
        raise ValueError(f"start_test ({start_test_main}) exceeds available data")

    #### output directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = Path(PROJECT_ROOT) / "results" / fund_name / "-".join(proxy_cols_main) / "RW_EQUI"
    output_dir.mkdir(parents=True, exist_ok=True)

    #### run simulations
    if n_threads == 1:
        results = []
        for i in range(n_sim):
            init_thread(Proxies_main, Fund_main, Proxies_train_main, Fund_train_main, 
                       proxy_cols_main, window_size_main, testing_main, start_test_main, 
                       start_train_main, r_main, file, fund_name)
            result = run_single_simulation(i)
            results.append(result)
            percentage = ((i + 1) / n_sim) * 100
            print(f"\rProgress: {i + 1}/{n_sim} ({percentage:.1f}%)", end='', flush=True)
    else:
        with Pool(processes=n_threads, initializer=init_thread, 
                  initargs=(Proxies_main, Fund_main, Proxies_train_main, Fund_train_main,
                           proxy_cols_main, window_size_main, testing_main, start_test_main,
                           start_train_main, r_main, file, fund_name)) as pool:
            results = []
            for i, result in enumerate(pool.imap(run_single_simulation, range(n_sim)), 1):
                results.append(result)
                percentage = (i / n_sim) * 100
                print(f"\rProgress: {i}/{n_sim} ({percentage:.1f}%)", end='', flush=True)

    print()

    finals = np.array([r[0] for r in results])
    avte = np.array([r[1] for r in results])

    #### plot final day tracking error
    mean_final, std_final = finals.mean(), finals.std()
    x_min, x_max = mean_final - 3*std_final, mean_final + 3*std_final

    bins = 50
    plt.figure(figsize=(8, 6))
    plt.hist(finals, bins=np.linspace(x_min, x_max, bins), edgecolor='black', alpha=0.7, density=True)
    plt.xlabel("Final Day Tracking Error", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distribution of Final Tracking Errors (Rolling Regression)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(mean_final, color='red', linestyle='dashed', linewidth=2, label=f"Mean={mean_final:.4f}")
    plt.legend(fontsize=10)
    plt.tight_layout()

    plot_file = output_dir / "RR_final_tracking_error.png"
    plt.savefig(plot_file, dpi=200)
    plt.show()

    #### plot average tracking error
    mean_avg, std_avg = avte.mean(), avte.std()
    x_min, x_max = mean_avg - 3*std_avg, mean_avg + 3*std_avg

    plt.figure(figsize=(8, 6))
    plt.hist(avte, bins=np.linspace(x_min, x_max, bins), edgecolor='black', alpha=0.7, density=True)
    plt.xlabel("Average Tracking Error", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distribution of Average Tracking Errors (Rolling Regression)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(mean_avg, color='red', linestyle='dashed', linewidth=2, label=f"Mean={mean_avg:.4f}")
    plt.legend(fontsize=10)
    plt.tight_layout()

    plot_file = output_dir / "RR_average_tracking_error.png"
    plt.savefig(plot_file, dpi=200)
    plt.show()

    #### save results
    results_df = pd.DataFrame({
        'Final_Tracking_Error': finals,
        'Average_Tracking_Error': avte
    })

    excel_file = output_dir / "RR_sim_results_QQQxEQUI.xlsx"
    results_df.to_excel(excel_file, index=False)