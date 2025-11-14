import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Hedging import ProxyHedge_self, Hedge_F
from Pricing import Price, Price_F
from utils import read_xlsx_file


class EWMAModel:
    def __init__(self, data_file, fund_name, proxy_cols, alpha, window_size, start_day, 
                 testing_size, r, volatility_calibration_days=400):
        self.data = read_xlsx_file(data_file)
        self.fund_name = fund_name
        self.proxy_cols = proxy_cols
        self.alpha = alpha
        self.window_size = window_size
        self.start_day = start_day
        self.testing_size = testing_size
        self.r = r
        self.volatility_calibration_days = volatility_calibration_days

        self.assets = self.data[self.proxy_cols].to_numpy()
        self.fund = self.data[self.fund_name].to_numpy()
        self.ewma_weights = None

    def calibrate_ewma_weights(self):

        #### initialize EWMA weights using OLS on calibration window
        calib_start = self.start_day - self.window_size
        assets_calib = self.assets[calib_start:self.start_day, :]
        fund_calib = self.fund[calib_start:self.start_day]

        ols_model = LinearRegression(fit_intercept=False)
        ols_model.fit(assets_calib, fund_calib)
        self.ewma_weights = ols_model.coef_.copy()

    def update_ewma_weights(self, weights_ols):

        #### update EWMA weights with exponential smoothing
        if self.ewma_weights is None:
            raise ValueError("EWMA weights not initialized")

        self.ewma_weights = self.alpha * weights_ols + (1 - self.alpha) * self.ewma_weights
        return self.ewma_weights.copy()

    def fit_OLS(self, assets_window, fund_window):
        ols_model = LinearRegression(fit_intercept=False)
        ols_model.fit(assets_window, fund_window)
        return ols_model.coef_

    def compute_rolling_weights(self):
        
        #### compute EWMA weights for each day using exponential smoothing
        self.calibrate_ewma_weights()
        rolling_weights = []

        for i in range(self.testing_size):
            current_day = self.start_day + i
            train_start = current_day - self.window_size
            train_end = current_day

            assets_train = self.assets[train_start:train_end, :]
            fund_train = self.fund[train_start:train_end]

            weights_ols = self.fit_OLS(assets_train, fund_train)

            if i > 0:
                current_weights = self.update_ewma_weights(weights_ols)
            else:
                current_weights = self.ewma_weights.copy()

            rolling_weights.append(current_weights)

        return np.array(rolling_weights)

    def compute_volatility_params(self):
        #### calibrate volatility using fixed window before testing period
        train_end = self.start_day
        train_start = self.start_day - self.volatility_calibration_days

        assets_train = self.assets[train_start:train_end, :]
        fund_train = self.fund[train_start:train_end]

        delta = np.diff(assets_train, axis=0)
        sigma_annual = delta.std(axis=0, ddof=1) * np.sqrt(252)
        corr = np.corrcoef(delta, rowvar=False)

        delta_y = np.diff(fund_train)
        sigma_fund = delta_y.std() * np.sqrt(252)

        return sigma_annual, corr, sigma_fund

    def run_hedge(self, Strike):
        assets_test = self.assets[self.start_day:self.start_day + self.testing_size, :].T
        fund_test = self.fund[self.start_day:self.start_day + self.testing_size]

        weights = self.compute_rolling_weights()
        sigma_annual, corr, sigma_fund = self.compute_volatility_params()

        #### proxy hedge with EWMA weights
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

        #### liability for comparison
        tracking_errors2, _, V, _ = Hedge_F(
            Strike=Strike,
            fund=fund_test,
            sigma=sigma_fund,
            r=self.r,
            T=self.testing_size,
            Price_F=Price_F
        )

        replication = np.array([
            np.dot(weights[i], assets_test[:, i])
            for i in range(self.testing_size)
        ])

        return {
            "tracking_errors": tracking_errors,
            "hedge_values": Hedge,
            "liability_values": V,
            "gradients": grads,
            "weights": weights,
            "replication": replication,
            "fund_actual": fund_test
        }