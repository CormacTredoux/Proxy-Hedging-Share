import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import read_xlsx_file


class StaticBlackBoxFund:
    #### creates static weighted fund with constant weights from day 1
    def __init__(self, data_file, seed=None, F0_index=520, standard_value=300):
        np.random.seed(seed)
        self.data_file = data_file
        self.data = read_xlsx_file(data_file)
        
        numeric_data = self.data.select_dtypes(include=[np.number]).copy()
        
        self.proxy_data = numeric_data.iloc[:, :15]
        self.assets = self.proxy_data.columns.tolist()
        self.n_assets = len(self.assets)
        
        #### filter max 7 indices (start with 'M')
        indices_candidates = [c for c in self.assets if c.startswith('M') and c not in ['META', 'QQQ', 'SPDR']]
        self.indices = indices_candidates[:7]
        
        self.categories = ['main', 'sec', 'ter']
        n = self.n_assets
        self.cat_idx = {
            'main': list(range(0, n//3)),
            'sec': list(range(n//3, 2*n//3)),
            'ter': list(range(2*n//3, n))
        }
        
        self.weight_bounds = {
            'main': (0.05, 0.8),
            'sec': (0.02, 0.15),
            'ter': (0.0, 0.05)
        }
        
        self.static_weights = self._initialize_static_weights()
        self.fund_values = self._compute_fund_values()
        self.fund_values = self._standardize_fund(F0_index, standard_value)
    
    def _initialize_static_weights(self):
        #### force MXEA = 50%, distribute remaining 50% across other assets
        w0 = np.zeros(self.n_assets)

        if 'MXEA' in self.assets:
            mxea_idx = self.assets.index('MXEA')
            w0[mxea_idx] = 0.5
        
        remaining_weight = 1.0 - w0.sum()
        remaining_idxs = [i for i in range(self.n_assets) if w0[i] == 0]

        for cat in self.categories:
            idxs = [i for i in remaining_idxs if i in self.cat_idx[cat]]
            if not idxs:
                continue
            w_cat = np.random.dirichlet(np.ones(len(idxs))) * remaining_weight
            min_w, max_w = self.weight_bounds[cat]
            w_cat = np.clip(w_cat, min_w, max_w)
            w0[idxs] = w_cat

        w0 /= w0.sum()

        return w0
    
    def _compute_fund_values(self):
        #### compute fund values using static weights
        weighted_sum = np.sum(self.proxy_data.to_numpy() * self.static_weights, axis=1)
        return pd.Series(weighted_sum, index=self.proxy_data.index)
    
    def _standardize_fund(self, F0_index, standard_value):
        #### scale fund so that fund value at F0_index equals standard_value
        scale_factor = standard_value / self.fund_values.iloc[F0_index]
        return self.fund_values * scale_factor

    def get_weights_df(self):
        return pd.DataFrame({
            'Asset': self.assets,
            'Weight': self.static_weights,
            'Category': [self._get_category(i) for i in range(self.n_assets)]
        }).sort_values('Weight', ascending=False)
    
    def _get_category(self, idx):
        for cat, idxs in self.cat_idx.items():
            if idx in idxs:
                return cat
        return 'unknown'
    
    def plot_weights(self, top_n=20):
        #### bar chart of top N weights
        weights_df = self.get_weights_df().head(top_n)
        color_map = {'main': 'steelblue', 'sec': 'purple', 'ter': 'orange'}
        colors = [color_map[cat] for cat in weights_df['Category']]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(weights_df)), weights_df['Weight'], color=colors)
        ax.set_xticks(range(len(weights_df)))
        ax.set_xticklabels(weights_df['Asset'], rotation=45, ha='right')
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title(f'Static BBF Weights (Top {top_n})', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['main'], label='Main (5-80%)'),
            Patch(facecolor=color_map['sec'], label='Secondary (2-15%)'),
            Patch(facecolor=color_map['ter'], label='Tertiary (0-5%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.show()
        return weights_df
    
    def save_to_excel(self, output_file='Combined_data_new.xlsx'):
        #### save BBF values to excel
        existing_data = read_xlsx_file(self.data_file)
        existing_data['BBF'] = self.fund_values.values
        
        script_dir = Path(__file__).resolve().parent.parent
        clean_dat_dir = script_dir / 'Clean_Dat'
        clean_dat_dir.mkdir(parents=True, exist_ok=True)
        output_path = clean_dat_dir / output_file
        existing_data.to_excel(output_path, index=False)
        
        return output_path


def main():
    bbf = StaticBlackBoxFund(
        data_file='Combined_data_new.xlsx',
        seed=42,
        F0_index=600,
        standard_value=300
    )
    
    weights_df = bbf.get_weights_df()
    bbf.plot_weights(top_n=20)
    
    #### plot fund values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bbf.fund_values.index, bbf.fund_values.values, linewidth=2, color='steelblue')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Fund Value', fontsize=12)
    ax.set_title('Black Box Fund (BBF)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    output_path = bbf.save_to_excel('Combined_data_new.xlsx')
    
    return bbf


if __name__ == '__main__':
    bbf = main()