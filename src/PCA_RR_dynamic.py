import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Hedging import ProxyHedge_self, Hedge_F
from Pricing import Price, Price_F
from utils import read_xlsx_file
from PCA_util import rolling_pca


class RollingRegressionPCAModel:
    #### dynamic rolling regression with PCA-driven proxy selection
    def __init__(self, data_file, fund_name, window_size, start_day, testing_size, r, volatility_calibration_days=400):
        self.data = read_xlsx_file(data_file)
        self.fund_name = fund_name
        self.window_size = window_size
        self.start_day = start_day
        self.testing_size = testing_size
        self.r = r
        self.volatility_calibration_days = volatility_calibration_days
        
        #### all potential proxy columns
        all_potential_cols = [
            "M1US000G","M6US0IN","MXEA","MXEF","MXEU",
            "MXUS","MXUS0CD","MXUS0IT","MXWD","MXWO",
            "MXWO000G","MXWO0IT","QQQ", "BBF"
        ]
        self.all_proxy_cols = [c for c in all_potential_cols if c != self.fund_name]
        
        self.fund = self.data[self.fund_name].to_numpy()
        self.assets_full = self.data[self.all_proxy_cols].to_numpy()
        
        pca_input_df = self.data[[self.fund_name] + self.all_proxy_cols]
        self.pca_selection = rolling_pca(pca_input_df, fund_col=fund_name)
    
    def compute_rolling_weights(self):
        #### compute rolling OLS weights using PCA-selected proxies
        rolling_weights = []

        pca_start_index = self.pca_selection.index.get_loc(self.data.index[self.window_size])
        
        for i in range(self.testing_size):
            current_day_in_test = i
            
            selection_row = self.pca_selection.iloc[pca_start_index + current_day_in_test]
            top3 = selection_row[selection_row == 1].index.tolist()
            
            selected_indices = [self.all_proxy_cols.index(a) for a in top3]
            
            train_start = self.start_day + i - self.window_size
            train_end = self.start_day + i
            X_train = self.assets_full[train_start:train_end][:, selected_indices]
            y_train = self.fund[train_start:train_end]
            
            model = LinearRegression(fit_intercept=False)
            model.fit(X_train, y_train)
            coef = model.coef_
            
            w_full = np.zeros(len(self.all_proxy_cols))
            for idx, val in zip(selected_indices, coef):
                w_full[idx] = val
            rolling_weights.append(w_full)
            
        return np.array(rolling_weights)
    
    def compute_volatility_params(self):
        #### calibrate volatility using fixed window before testing period
        train_start = self.start_day - self.volatility_calibration_days
        train_end = self.start_day
        
        delta = np.diff(self.assets_full[train_start:train_end], axis=0)
        sigma_annual = delta.std(axis=0, ddof=1) * np.sqrt(252)
        corr = np.corrcoef(delta, rowvar=False)
        
        delta_f = np.diff(self.fund[train_start:train_end])
        sigma_fund = delta_f.std() * np.sqrt(252)
        
        return sigma_annual, corr, sigma_fund
    
    def run_hedge(self, Strike):
        assets_test = self.assets_full[self.start_day:self.start_day + self.testing_size, :].T
        fund_test = self.fund[self.start_day:self.start_day + self.testing_size]
        
        weights = self.compute_rolling_weights()
        sigma_annual, corr, sigma_fund = self.compute_volatility_params()
        
        #### proxy hedge with PCA-selected weights
        tracking_errors, Hedge, Vp, grads = ProxyHedge_self(
            Strike=Strike,
            assets=assets_test,
            weights=weights,
            sigma_annual=sigma_annual,
            corr=corr,
            r=self.r,
            T=self.testing_size,
            Price=Price,
            sigma_fund=sigma_fund,
            fund=fund_test
        )
        
        #### perfect liability for comparison
        tracking_errors2, _, V, _ = Hedge_F(
            Strike=Strike,
            fund=fund_test,
            sigma=sigma_fund,
            r=self.r,
            T=self.testing_size,
            Price_F=Price_F
        )
        
        replication = np.array([np.dot(weights[i], assets_test[:, i]) for i in range(self.testing_size)])
        
        return {
            "tracking_errors": tracking_errors,
            "hedge_values": Hedge,
            "liability_values": V,
            "gradients": grads,
            "weights": weights,
            "replication": replication,
            "fund_actual": fund_test
        }