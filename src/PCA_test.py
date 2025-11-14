import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from PCA_util import rolling_pca
from utils import read_xlsx_file, format_time_axis, finalize_plot


#### config
file = "Combined_data_new.xlsx"
cols = ["M1US000G","M6US0IN","MXEA","MXEF","MXEU",
        "MXUS","MXUS0CD","MXUS0IT","MXWD","MXWO",
        "MXWO000G","MXWO0IT","QQQ", "BBF"]
fund_col = "BBF"


data = read_xlsx_file(file)

try:
    all_dates = pd.to_datetime(data.iloc[:, 0].values)
except Exception as e:
    all_dates = pd.to_datetime(pd.RangeIndex(start=0, stop=len(data), step=1), unit='D')

data = data[cols].copy()
data = data.sort_index()

#### compute indicators using rolling PCA
delayed_indicator_df = rolling_pca(
    df=data,
    fund_col=fund_col,
    rolling_window=50,
    k_pcs_for_similarity=3,
    top_n=3,
    drop_days=7
)

assets_to_consider = [a for a in data.columns if a != fund_col]

#### clip data to days 600-852
clip_start, clip_end = 600, 852
delayed_indicator_df_clip = delayed_indicator_df.iloc[clip_start:clip_end]

if len(all_dates) < clip_end:
     raise IndexError(f"Data length ({len(all_dates)}) insufficient for clip_end ({clip_end})")

#### plot delayed top-n indicator
fig, ax = plt.subplots(figsize=(12, 6))

for i, asset in enumerate(assets_to_consider):
    clip_indices = np.where(delayed_indicator_df_clip[asset].values == 1)[0]
    days = clip_indices + clip_start
    ax.vlines(days, i - 0.4, i + 0.4, color='black', linewidth=3.4)

ax.set_yticks(range(len(assets_to_consider)))
ax.set_yticklabels(assets_to_consider)
ax.set_ylim(-0.5, len(assets_to_consider) - 0.5)

finalize_plot(ax, xlabel="Date", ylabel="Asset", legend=False)

#### custom x-axis date labels
num_ticks = 7
tick_positions = np.linspace(clip_start, clip_end, num_ticks, dtype=int)

try:
    tick_dates = all_dates[tick_positions]
except IndexError:
    tick_positions = tick_positions[tick_positions < len(all_dates)]
    tick_dates = all_dates[tick_positions]
    
tick_labels = [date.strftime('%d/%b') for date in tick_dates]

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0)

#### add year labels
transform = ax.get_xaxis_transform()
y_text_pos = -0.15

years_in_clip = all_dates[clip_start:clip_end].year.unique()
first_year = all_dates[clip_start].year

for year in years_in_clip:
    if year == first_year:
        year_start_index = clip_start
        ha = 'left'
    else:
        year_start_date = pd.Timestamp(year, 1, 1)
        year_start_index = all_dates.searchsorted(year_start_date)
        year_start_index = max(year_start_index, clip_start)
        ha = 'center'
        
    if year_start_index <= clip_end:
         ax.text(year_start_index, y_text_pos, str(year),
                 transform=transform, ha=ha, va='top', fontsize=10)

ax.set_xlim(clip_start, clip_end)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2) 
plt.show()