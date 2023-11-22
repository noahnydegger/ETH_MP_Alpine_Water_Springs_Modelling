#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:59:13 2023

@author: ramunbar
"""

# use this link https://lmc2179.github.io/posts/autoreg.html
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib.dates import MonthLocator, DateFormatter
from statsmodels.tsa.stattools import adfuller

# use this link https://lmc2179.github.io/posts/autoreg.html
# useful link: https://www.statsmodels.org/dev/examples/notebooks/generated/predict.html
# https://www.statsmodels.org/dev/examples/notebooks/generated/autoregressions.html

#Path to used dataset
data_path = Path(__file__).parent / ".." / "Data/spring_data/resampled_data/Ulrika/Ulrika_H.csv"

def get_ulrika(show_plot):
    ulrika = pd.read_csv(data_path)
    # Convert 'datetime' column to datetime object()
    ulrika['datetime'] = pd.to_datetime(ulrika['datetime'], utc=True)
    # Filter and create ulrika_d dataframe
    ulrika_d = ulrika.loc[(ulrika['discharge(L/min)'] > 0) & (ulrika['discharge(L/min)'] <= 2000)].copy()
    # Create a figure
    if show_plot:
        fig, ax_flow = plt.subplots(figsize=(15, 9))

        # Plot the spring data
        ax_flow.plot(ulrika_d.datetime, ulrika_d['discharge(L/min)'], linewidth=1, color="blue",
                     label='spring discharge', zorder=1)
        plt.ylabel('Discharge [l/min]')
        # Set the x-axis ticks to display a subset of dates and rotate them by 45 degrees
        x_ticks = ulrika_d.datetime[::1440]  # Adjust the interval as needed
        ax_flow.set_xticks(x_ticks)
        ax_flow.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.title('Filtered Dataframe (ulrika_d)')

    return ulrika_d


df = get_ulrika(False)
df = df.dropna()

discharge = df["discharge(L/min)"]
df = pd.DataFrame({

    'discharge': df['discharge(L/min)']
})

pacf = plot_pacf(df.discharge, lags=100)
# Check for stationarity of the time-series data
# In case, p-value is less than 0.05, the time series
# data can said to have stationarity

stationarityTest = adfuller(discharge, autolag='AIC')
#
# Check the value of p-value
#
print("P-value: ", stationarityTest[1])
#
# Next step is to find the order of AR model to be trained
# for this, we will plot partial autocorrelation plot to assess
# the direct effect of past data on future data
#
pacf = plot_pacf(discharge, lags=25)

df['dlog'] = np.log(df.discharge)
df['t'] = range(len(df))

# split data
trn_end = round(np.size(df.discharge) / 4)
trn = df.iloc[:trn_end]['discharge'].dropna()
tst = df.iloc[trn_end:]['discharge'].dropna()

# Fit using statsmodel
l = np.array([1, 3, 4, 5])
mod = AutoReg(trn, l, old_names=False)
res = mod.fit()
print(res.summary())

data_ = res.predict(end=np.size(df.index) - 1, dynamic=trn_end)

fig, ax = plt.subplots()
ax.plot(df.t, df.discharge, "-", label="latent")
ax.plot(df.t[:trn_end], trn, "o", label="training")
ax.plot(df.t[trn_end:], tst, "x", label="testing")
ax.plot(data_.index, data_, "--", label="model")
plt.legend()

# %%
train_cutoff = np.size(df) / 3
validate_cutoff = np.size(df) / 3 * 2

train_df = df[df['t'] <= train_cutoff]
select_df = df[(df['t'] > train_cutoff) & (df['t'] <= validate_cutoff)]
forecast_df = df[df['t'] > validate_cutoff]

plt.plot(train_df.t, train_df.discharge, label='Training data')
plt.plot(select_df.t, select_df.discharge, label='Model selection holdout')
plt.legend()
plt.title('Data overview')
plt.ylabel('discharge')
plt.xlabel('Index')

plt.show()