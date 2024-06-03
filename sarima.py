import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import aic, bic
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.statespace.sarimax import SARIMAX


def LSE(X, y):
    if y.ndim != 1:
        raise ValueError("y is not 1d")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y lengths are different")

    X = np.column_stack((np.ones(X.shape[0]), X))
    coeffs = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    return coeffs


def difference(timeseries, lag=1):
    return np.array([timeseries[i] - timeseries[i - lag] for i in range(lag, len(timeseries))])


def reverse_difference(differenced, lag, initial_values):
    reconstructed = list(initial_values)

    for value in differenced:
        reconstructed.append(reconstructed[-lag] + value)

    return reconstructed


def timeseries_to_matrix(ts, p, P, s):
    X, y = [], []

    max_lag = max(p, P * s)
    for i in range(max_lag, len(ts)):
        window = list(ts[i - p:i])

        for j in range(1, P + 1):
            window.append(ts[i - j * s])

        X.append(window)
        y.append(ts[i])

    return np.array(X), np.array(y)


class SARIMA:
    def __init__(self, ts, p, d, q, P, D, Q, s):
        self.retrieval_info = []
        diff_ts = np.array(ts.copy(deep=True))
        for i in range(D):
            self.retrieval_info.append(diff_ts[len(diff_ts) - s:])
            diff_ts = difference(diff_ts, s)
        for i in range(d):
            self.retrieval_info.append([diff_ts[-1]])
            diff_ts = difference(diff_ts)

        self.ts = diff_ts
        self.residuals = np.empty(len(self.ts))
        self._p = p
        self._d = d
        self._q = q
        self._P = P
        self._D = D
        self._Q = Q
        self._seasonality = s

        self.ar_coefs = None
        self.sar_coefs = None
        self.ar_sar_intercept = None

        self.ma_coefs = None
        self.sma_coefs = None
        self.ma_sma_intercept = None

        self.aic = None
        self.bic = None
        self.hqic = None
        self.mse = None

    def _fit_SAR_AR(self, end):
        X, y = timeseries_to_matrix(self.ts[:end + 1], self._p, self._P, self._seasonality)
        lr = LinearRegression()
        lr.fit(X, y)
        coefs = lr.coef_

        self.ar_coefs, self.sar_coefs, self.ar_sar_intercept = coefs[:self._p], coefs[
                                                                                self._p:self._p + self._P], lr.intercept_
        return self.ar_coefs, self.sar_coefs, self.ar_sar_intercept

    def _predict_SAR_AR(self, sample):
        # if sample.shape[0] != self._p + self._P:
        #     raise ValueError(f"The sample length {len(sample)} is not equal to {self._p + self._P}")
        predict = sample.dot(np.concatenate((self.ar_coefs, self.sar_coefs))) + self.ar_sar_intercept
        return predict

    def _get_residuals(self, end):
        X, y = timeseries_to_matrix(self.ts[:end + 1], self._p, self._P, self._seasonality)
        predictions = self._predict_SAR_AR(X)
        self.residuals[-len(y):] = y - predictions

        return y - predictions

    def _fit_MA_SMA(self, residuals):
        X, y = timeseries_to_matrix(residuals, self._q, self._Q, self._seasonality)
        lr = LinearRegression()
        lr.fit(X, y)
        coefs = lr.coef_
        self.ma_coefs, self.sma_coefs, self.ma_sma_intercept = coefs[:self._q], coefs[
                                                                                self._q:self._q + self._Q], lr.intercept_
        return self.ma_coefs, self.sma_coefs, self.ma_sma_intercept

    # def predict_MA_SMA(self, residuals):
    #     if residuals.shape[0] != self._q + self._Q:
    #         raise ValueError(f"The residuals length {len(residuals)} is not equal to {self._q + self._Q}")
    #     predict = residuals.dot(np.concatenate((self.ma_coefs, self.sma_coefs))) + self.ma_sma_intercept
    #     return predict

    # def predict(self, start, end):
    #     forecasts = []
    #     ts_extended = list(self.ts_train+ self.ts_test)
    #     for i in range(start, end + 1):
    #         ar_sar_sample = np.array(
    #             ts_extended[i - self._p:i] + [ts_extended[i - j * self._seasonality] for j in range(1, self._P + 1)])
    #         ar_sar_pred = self.predict_SAR_AR(ar_sar_sample)
    #
    #         # residuals = self.get_residuals(ts_extended[:i])
    #
    #         ma_sma_sample = self.residuals[-(self._q + self._Q):]
    #         ma_sma_pred = self.predict_MA_SMA(ma_sma_sample)
    #
    #         final_pred = ar_sar_pred + ma_sma_pred
    #
    #         forecasts.append(final_pred)
    #         ts_extended.append(final_pred)
    #     return forecasts

    def fit(self, end_t):
        self._fit_SAR_AR(end_t)
        resid = self._get_residuals(end_t)
        self._fit_MA_SMA(resid)

    def predict_in_sample(self, start, end):
        start -= (self._D * self._seasonality + self._d)
        end -= (self._D * self._seasonality + self._d)
        self.fit(end)
        forecasts = []

        ts = deepcopy(self.ts)
        residuals = deepcopy(self.residuals)
        # if end > len(ts):
        #     raise ValueError("Can't forecast more than in_sample")

        if start - max(self._q, self._Q * self._seasonality) < 0 or start - max(self._p,
                                                                                self._P * self._seasonality) < 0:
            raise ValueError(f"Can't get enough values for start point {start}. The start is \"too early\"")

        for i in range(start, end + 1):
            if i < len(self.ts):
                ma = residuals[i - self._q:i].dot(self.ma_coefs)
                sma = np.array([residuals[i - j * self._seasonality] for j in range(self._Q, 0, -1)]).dot(
                    self.sma_coefs)
                ar = ts[i - self._p:i].dot(self.ar_coefs)
                sar = np.array([ts[i - j * self._seasonality] for j in range(self._P, 0, -1)]).dot(
                    self.sar_coefs)
            elif i >= len(self.ts):
                ma = residuals[-q:].dot(self.ma_coefs)
                sma = np.array([residuals[- j * self._seasonality] for j in range(self._Q, 0, -1)]).dot(
                    self.sma_coefs)
                ar = ts[-self._p:].dot(self.ar_coefs)
                sar = np.array([ts[- j * self._seasonality] for j in range(self._P, 0, -1)]).dot(
                    self.sar_coefs)

            forecast = ar + sar + self.ar_sar_intercept + ma + sma + self.ma_sma_intercept
            forecasts.append(forecast)
            if i >= len(self.ts):
                np.append(ts, forecast)
            else:
                residuals[i] = ts[i] - forecast

        for r_i in reversed(self.retrieval_info):
            if len(r_i) == 1:
                forecasts = np.concatenate((r_i, forecasts))
                forecasts = np.cumsum(forecasts)
                forecasts = forecasts[1:]
            else:
                forecasts = np.concatenate((r_i, forecasts))
                for i in range(self._seasonality, len(forecasts)):
                    forecasts[i] += forecasts[i - self._seasonality]
                forecasts = forecasts[self._seasonality:]

        if end >= len(ts):
            return forecasts

        errors = np.array(self.ts[start:end + 1]) - forecasts

        self.mse = np.mean(errors ** 2)

        n = len(errors)
        sigma2 = np.var(errors)
        log_likelihood = -n / 2 * (np.log(2 * np.pi) + 2*np.log(sigma2)) - np.sum(errors**2/(2*sigma2**2))

        k = self._p + self._P + self._q + self._Q + 2

        self.aic = 2 * k - 2 * log_likelihood
        self.bic = np.log(n) * k - 2 * log_likelihood
        self.hqic = -2 * log_likelihood + 2 * k * np.log(np.log(n))

        return forecasts


if __name__ == '__main__':
    df = pd.read_csv("monthly-beer-production-in-austr.csv")
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    df.index.freq = 'MS'
    ts = df["Monthly beer production"]

    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    p, d, q, P, D, Q, s = 3, 0, 3, 0, 1, 1, 12
    # p, d, q, P, D, Q, s = 1, 2, 1, 2, 1, 1, 12
    my_model = SARIMA(ts, p, d, q, P, D, Q, s)
    my_forecast = my_model.predict_in_sample(200, 224)

    sm_model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
    sm_results = sm_model.fit(disp=False)
    sm_forecast = sm_results.predict(start=200, end=224, dynamic=True)

    print(len(train[100:len(train)]))
    print(len(my_forecast))

    my_mae = mean_squared_error(ts[200:225], my_forecast)
    sm_mae = mean_squared_error(ts[200:225], sm_forecast)

    print(f'My SARIMA MSE: {my_mae}, AIC: {my_model.aic}, BIC: {my_model.bic}, HQIC: {my_model.hqic}')
    print(f'SARIMAX MSE: {sm_mae}, AIC: {sm_results.aic}, BIC: {sm_results.bic}, HQIC: {sm_results.hqic}')

    # Visualization
    plt.figure(figsize=(15, 6))
    plt.plot(ts.index[200:225], ts[200:225], label='Actual', color='black')
    plt.plot(ts.index[200:225], my_forecast, label='My SARIMA Forecast', color='red')
    plt.plot(ts.index[200:225], sm_forecast, label='statsmodels SARIMAX Forecast', color='blue')
    plt.legend(loc='best')
    plt.title('SARIMA Forecast Comparison')
    plt.show()
# df = pd.read_csv("monthly-beer-production-in-austr.csv")
# df['Month'] = pd.to_datetime(df['Month'])
# df.set_index('Month', inplace=True)
# df.index.freq = 'MS'
# ts = df["Monthly beer production"]
#
# train_size = int(len(ts) * 0.8)
# train, test = ts[:train_size], ts[train_size:]
#
# p, d, q, P, D, Q, s = 3, 0, 3, 0, 1, 1, 12
#
# my_model = SARIMA(ts, p, d, q, P, D, Q, s)
# my_forecast = my_model.predict_in_sample(len(ts), len(ts) + 24)
#
# sm_model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
# sm_results = sm_model.fit(disp=False)
# sm_forecast = sm_results.predict(start=len(ts), end=len(ts) + 24, dynamic=True)
#
# print(len(ts))
# print(len(my_forecast))
#
# my_mae = mean_absolute_error(ts[200:225], my_forecast)
# sm_mae = mean_absolute_error(ts[200:225], sm_forecast)
#
# print(f'My SARIMA MАE: {my_mae}, AIC: {my_model.aic}, BIC: {my_model.bic}, HQIC: {my_model.hqic}')
# print(f'SARIMAX MАE: {sm_mae}, AIC: {sm_results.aic}, BIC: {sm_results.bic}, HQIC: {sm_results.hqic}')
#
#
# forecast_index = pd.date_range(start=ts.index[-1], periods=25, freq='M')
#
# # Visualization
# plt.figure(figsize=(15, 6))
# #plt.plot(ts.index, ts, label='Actual', color='black')
# plt.plot(forecast_index, my_forecast, label='My SARIMA Forecast', color='red')
# plt.plot(forecast_index, sm_forecast, label='statsmodels SARIMAX Forecast', color='blue')
# plt.legend(loc='best')
# plt.title('SARIMA Forecast Comparison')
# plt.show()
