import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm


def rolling_pca(df, fund_col, rolling_window=50, k_pcs_for_similarity=3, top_n=3, drop_days=7):
    #### compute post-hoc delayed top-n PCA indicator for fund vs assets
    df = df.copy().sort_index()
    
    returns = np.log(df / df.shift(1)).dropna(how="any")
    assets = [a for a in df.columns if a != fund_col]

    def run_pca_on_window(window_returns):
        #### PCA on one window
        X = StandardScaler().fit_transform(window_returns)
        pca = PCA()
        pca.fit(X)
        loadings = pd.DataFrame(
            pca.components_.T,
            index=window_returns.columns,
            columns=[f"PC{i+1}" for i in range(len(window_returns.columns))]
        )
        return loadings

    #### rolling PCA and top-n selection
    indicator_list = []

    for i in range(rolling_window, len(returns)):
        window = returns.iloc[i-rolling_window:i]
        date = returns.index[i]

        loadings = run_pca_on_window(window)
        use_cols = [f"PC{i+1}" for i in range(min(k_pcs_for_similarity, loadings.shape[1]))]

        #### cosine similarity vs fund
        fund_vector = loadings.loc[fund_col, use_cols].values
        similarities = {}
        for col in assets:
            v = loadings.loc[col, use_cols].values
            similarities[col] = float(np.dot(fund_vector, v) / (norm(fund_vector) * norm(v)))

        top_assets = sorted(similarities, key=similarities.get, reverse=True)[:top_n]

        indicator_row = pd.Series(0, index=df.columns, name=date)
        for asset in top_assets:
            indicator_row[asset] = 1
        indicator_list.append(indicator_row)

    indicator_df = pd.DataFrame(indicator_list)

    #### post-hoc delayed drop indicator with slotting
    delayed_indicator_df = pd.DataFrame(0, index=indicator_df.index, columns=indicator_df.columns)

    first_row_top_assets = indicator_df.iloc[0][assets].sort_values(ascending=False).head(top_n).index.tolist()
    current_top = first_row_top_assets.copy()
    delayed_indicator_df.iloc[0, delayed_indicator_df.columns.get_indexer(current_top)] = 1

    for t in range(1, len(indicator_df)):
        new_top = current_top.copy()
        start_idx = max(0, t - drop_days)
        window = indicator_df.iloc[start_idx:t+1][assets]

        #### remove assets with [1,0,...,0] pattern
        for asset in current_top.copy():
            if t - drop_days >= 0:
                pattern = window[asset].values
                if np.array_equal(pattern, [1] + [0]*drop_days):
                    new_top.remove(asset)

        #### fill remaining slots
        n_needed = top_n - len(new_top)
        if n_needed > 0:
            scores = window.sum(axis=0)
            scores[new_top] = -1
            last_day = window.iloc[-1]
            scores += last_day * 0.01
            replacements = scores.sort_values(ascending=False).head(n_needed).index.tolist()
            new_top += replacements

        delayed_indicator_df.iloc[t, delayed_indicator_df.columns.get_indexer(new_top)] = 1
        current_top = new_top.copy()

    return delayed_indicator_df