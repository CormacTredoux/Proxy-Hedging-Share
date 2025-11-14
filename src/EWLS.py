import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from Hedging import ProxyHedge_self, Hedge_F
from Pricing import Price, Price_F
from utils import read_xlsx_file


class EWLSModel:
    def __init__(self, data_file, fund_name, proxy_cols, window_size, start_day, 
                 testing_size, r, beta=0.97, volatility_calibration_days=400):
        self.data = read_xlsx_file(data_file)
        self.fund_name = fund_name
        self.proxy_cols = proxy_cols
        self.window_size = window_size
        self.start_day = start_day
        self.testing_size = testing_size
        self.r = r
        self.beta = beta
        self.volatility_calibration_days = volatility_calibration_days

        self.assets = self.data[self.proxy_cols].to_numpy()
        self.fund = self.data[self.fund_name].to_numpy()

    def compute_rolling_weights(self):
        #### compute EWLS weights for each day using exponential weighting
        rolling_weights = []

        for i in range(self.testing_size):
            current_day = self.start_day + i
            train_end = current_day
            train_start = current_day - self.window_size

            assets_train = self.assets[train_start:train_end, :]
            fund_train = self.fund[train_start:train_end]

            weights = self.fit_EWLS(assets_train, fund_train)
            rolling_weights.append(weights)

        return np.array(rolling_weights)

    def fit_EWLS(self, assets, fund):
        #### exponentially weighted least squares regression
        n = len(fund)
        weights = self.beta ** np.arange(n - 1, -1, -1)

        EWLS_model = LinearRegression(fit_intercept=False)
        EWLS_model.fit(assets, fund, sample_weight=weights)

        return EWLS_model.coef_

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

        #### proxy hedge with EWLS weights
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