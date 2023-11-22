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

# %%

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

# %%
# Load data
# -----------
#Path to used dataset
data_path = Path(__file__).parent / ".." / "Data/spring_data/resampled_data/Ulrika/Ulrika_H.csv"

df = get_ulrika(False)

qcol = "discharge(L/min)"
data = df[[qcol]].copy()
# set timedelta index
data.index = df.datetime - df.datetime.min()
data.index.name = "time"


# %%
# Explore data
# ------------
# %%
# Check for stationarity of the time-series data
# In case, p-value is less than 0.05, the time series
# data can be said to be stationary

stationarityTest = adfuller(data, autolag='AIC')
print("P-value: ", stationarityTest[1])

# %%
# Auto-correlation
# *****************
# Next step is to find the order of AR model to be trained
# for this, we will plot partial autocorrelation plot to assess
# the direct effect of past data on future data
#
pacf = plot_pacf(data, lags=25)

# %%
# Fit model
# ----------
# transform data
data[f"log_{qcol}"] = np.log(df[qcol])
nsteps = df.shape[0]

# %%
# split data
trn_end = int(nsteps * 0.75)
trn = data[qcol].iloc[:trn_end]
tst = data[qcol].iloc[trn_end:]

# Fit using statsmodel
l = np.array([1, 2, 7, 9, 10, 11, 12])  # lags based on PACF
mod = AutoReg(trn.to_numpy(), l, old_names=False)
res = mod.fit()
print(res.summary())

data["model"] = res.predict(end=nsteps-1, dynamic=trn_end)

fig, ax = plt.subplots()
data[qcol].plot(ax=ax)
ax.plot(trn, "o", label="training", )
ax.plot(tst,  "x", label="testing",)
ax.plot(data.model, "--", label="model")
plt.legend()

# %%
# train_cutoff = np.size(df) / 3
# validate_cutoff = np.size(df) / 3 * 2
#
# train_df = df[df['t'] <= train_cutoff]
# select_df = df[(df['t'] > train_cutoff) & (df['t'] <= validate_cutoff)]
# forecast_df = df[df['t'] > validate_cutoff]
#
# plt.plot(train_df.t, train_df.discharge, label='Training data')
# plt.plot(select_df.t, select_df.discharge, label='Model selection holdout')
# plt.legend()
# plt.title('Data overview')
# plt.ylabel('discharge')
# plt.xlabel('Index')

plt.show()