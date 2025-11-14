# utils.py
import os
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

## note it's much easier to just load in excel with pandas to a df and then just use .toNumpy to turn it into an ndarray
def read_xlsx_file(file_name):  ### For bloomberg data
    data_folder = os.path.join(PROJECT_ROOT, "Clean_Dat") 
    file_path = os.path.join(data_folder, file_name)

    # Read Excel file
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f"Excel file '{file_name}' not found in {data_folder}")

def read_csv_file(file_name):  ### For Yuri's data
    data_folder = os.path.join(PROJECT_ROOT, "Data")
    file_path = os.path.join(data_folder, file_name)

    if os.path.exists(file_path):
        return pd.read_csv(file_path, skiprows=1)
    raise FileNotFoundError(f"{file_name} not found in {data_folder}")

def read_txt_file(file_name, **kwargs): 
    """
    For saved states. Histotrical, from initial code given to us... might be useful later
    """
    data_folder = os.path.join(PROJECT_ROOT, "Saved_States")
    file_path = os.path.join(data_folder, file_name)
    if os.path.exists(file_path):
        return np.loadtxt(file_path, **kwargs)
    raise FileNotFoundError(f"{file_name} not found in {data_folder}")

def readData(data, NumProxies, FundNo):
    X = data.iloc[:, 1:(1+NumProxies)]
    Xnames = X.columns
    Dates = data.iloc[:, 0]
    X = X.to_numpy().T
    Y = data.iloc[:, NumProxies+FundNo] * 100_000_000
    FundName = Y.name
    Y = np.array([[y for y in Y.to_numpy().T]])
    return X, Y, Xnames, FundName, Dates

def generate_initial_states(NumProxies, scale=1.0, overwrite=True):
    """
    Generates and (optionally) saves initial states. Will be tweaked as we go for deciding with more informed states.
    The scale param is just there in case there for some easy manual tuning. There might be some hyperparam tuning potentials with it, but not really a focus rn
    """

    state0 = np.zeros((NumProxies, 1))
    P0 = np.identity(NumProxies) * scale
    Q = np.identity(NumProxies) * scale
    R = np.array([[0.01]])

    if overwrite:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        save_dir = os.path.join(parent_dir, "Saved_States")
        os.makedirs(save_dir, exist_ok=True)

        np.savetxt(os.path.join(save_dir, "state0.txt"), state0)
        np.savetxt(os.path.join(save_dir, "P0.txt"), P0)
        np.savetxt(os.path.join(save_dir, "Q.txt"), Q)
        np.savetxt(os.path.join(save_dir, "R.txt"), R)

    return state0, P0, Q, R



### SIMULATION

def get_results_directory(fund_name, proxy_cols, model_name):
    """Generate and create results directory: results/FUNDNAME/proxy1-proxy2-proxy3/MODEL_NAME/"""
    sorted_proxies = sorted(proxy_cols)
    dir_name = '-'.join(sorted_proxies)
    results_path = os.path.join('results', fund_name, dir_name, model_name)
    os.makedirs(results_path, exist_ok=True)
    return results_path



### Plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def format_time_axis(ax, date_array, tick_count=5, year_labels=True):
    """
    Apply consistent date formatting and layout to plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to format.
    date_array : array-like
        Array or list of datetime objects for the x-axis.
    tick_count : int, optional
        Number of major ticks to show across the x-axis (default=5).
    year_labels : bool, optional
        Whether to display year indicators below the axis (default=True).
    """
    if len(date_array) == 0:
        raise ValueError("date_array is empty — cannot format axis.")

    # Ensure x-limits
    ax.set_xlim(date_array[0], date_array[-1])

    # Evenly spaced ticks across the available date range
    tick_idx = np.linspace(0, len(date_array) - 1, tick_count, dtype=int)
    tick_dates = np.array(date_array)[tick_idx]
    ax.set_xticks(tick_dates)

    # Date format for tick labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%b'))
    ax.minorticks_off()

    # Slight padding and rotation for readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=9)

    # Add subtle vertical grid lines at ticks
    ax.grid(True, axis='x', alpha=0.2, linestyle=':')

    # Optional year indicators under the axis
    if year_labels:
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        start_date = date_array[0]
        next_year_jan1 = datetime(start_date.year + 1, 1, 1)
        # Find the first trading day on/after Jan 1
        next_year = None
        for d in date_array:
            if d >= next_year_jan1:
                next_year = d
                break

        # Adjust subplot bottom margin to fit labels
        plt.subplots_adjust(bottom=0.22)
        y_text = -0.12
        ax.text(start_date, y_text, str(start_date.year),
                transform=ax.get_xaxis_transform(), ha='left', va='top', fontsize=10)
        if next_year:
            ax.text(next_year, y_text, str(next_year.year),
                    transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=10)

    # Make layout tight for export
    plt.tight_layout()


def finalize_plot(ax, title=None, xlabel='Time', ylabel='Value', legend=True):
    """
    Apply consistent labels, grid, and optional title to any plot.
    """
    if title:
        ax.set_title(title, fontsize=12, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if legend:
        ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
