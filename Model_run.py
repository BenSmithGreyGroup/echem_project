import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# def model_run(X, y, model, extraArgs):
#     cycle_max = 150
#     n = 2
#
#     var_error_df = pd.DataFrame(index=range(1, cycle_max + 1, 1), columns=range(1, cycle_max + 1, 1))
#     outlier_df = pd.DataFrame(index=range(1, cycle_max + 1, 1), columns=range(1, cycle_max + 1, 1))
#
#     for i in range(1, cycle_max + 1, 1):
#         for j in range(1, i, 1):
#             x = var_df.loc[(slice(None), j), f'{i}'].values
#             mean = sum(x) / len(x)
#             std = np.std(x)
#
#             x_non_outlier = x[abs(x - mean) < n * std]
#             #         y = cycle_life_df[abs(x - mean) < n*std]['Cycle Life'].apply(np.log)
#             y = cycle_life_df.apply(np.log)
#             x_scaler = StandardScaler()
#             y_scaler = StandardScaler()
#             reg = anotherfunc(*extraArgs)
#
#             X_train, X_test, y_train, y_test = train_test_split(
#                 x, y, test_size=0.33, random_state=42)
#
#             X_train = np.array(X_train).reshape(-1, 1)
#             X_test = np.array(X_test).reshape(-1, 1)
#             y_train = np.array(y_train).reshape(-1, 1)
#             y_test = np.array(y_test).reshape(-1, 1)
#
#             x_scaler.fit(X_train)
#             y_scaler.fit(y_train)
#
#             X_train_scaled = x_scaler.transform(X_train)
#             X_test_scaled = x_scaler.transform(X_test)
#             y_train_scaled = y_scaler.transform(y_train)
#
#             reg.fit(X_train_scaled, y_train_scaled)
#             y_scaled_pred = reg.predict(X_test_scaled)
#             y_pred = y_scaler.inverse_transform(y_scaled_pred)
#
#             var_error_df.loc[j, i] = mean_absolute_error(y_test, y_pred)
#             outlier_df.loc[j, i] = len(x) - len(x_non_outlier)

def test(X, X_test, y, y_test, model, **extraargs):
    reg = model(n_estimators=arg1, max_depth=arg2)
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    plt.plot(y_test, y_pred)
    plt.show()
    plt.close()
    return


x = np.linspace(0,10, 100).reshape(-1, 1)
y = np.linspace(10, 20, 100)

x_test = np.linspace(0,10,10).reshape(-1, 1)
y_test = np.linspace(10, 20, 10) + np.random.normal(0, 0.01, 10)


test(x, x_test, y, y_test, RandomForestRegressor, {'arg1':100, 'arg2': 10})