#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:59:13 2023

@author: ramunbar
"""

# use this link https://lmc2179.github.io/posts/autoreg.html

from patsy import dmatrix, build_design_matrices
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import sammhelper as sh
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
# import get_cleaned_data
from matplotlib.dates import MonthLocator, DateFormatter
from statsmodels.tsa.stattools import adfuller

from patsy import dmatrix, build_design_matrices


# use this link https://lmc2179.github.io/posts/autoreg.html
# useful link: https://www.statsmodels.org/dev/examples/notebooks/generated/predict.html
# https://www.statsmodels.org/dev/examples/notebooks/generated/autoregressions.html


def get_ulrika(show_plot):
    path = "/Users/ramunbar/Documents/Master/3_Semester/GITHUB/ETH_MP_Alpine_Water_Springs_Modelling/Data/spring_data/resampled_data/Ulrika/Ulrika_H.csv"
    ulrika = pd.read_csv(path)
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
# Assuming 'datetime' is the column containing timestamps
# df['datetime'] = pd.to_datetime(df['datetime'])
# df = df.set_index('datetime')


discharge = df["discharge(L/min)"]
df = pd.DataFrame({

    'discharge': df['discharge(L/min)']
})

# df = df.asfreq('H')

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
# df['year'] = df['Month'].apply(lambda x: int(x.split('-')[0]))
# df['month_number'] = df['Month'].apply(lambda x: int(x.split('-')[1]))
df['t'] = range(len(df))

### MAKE A TRY WITH NUMPY ARRAY #####
#################################################
# Convert the specified column to a NumPy array
# numpy_array = df['discharge'].values
# trn_end = round(np.size(numpy_array)/4)
# trn = numpy_array[:trn_end]
# tst = numpy_array[trn_end:]


# split data
trn_end = round(np.size(df.discharge) / 4)
trn = df.iloc[:trn_end]['discharge'].dropna()
tst = df.iloc[trn_end:]['discharge'].dropna()
# If this is done na goes to the data if its not done the autoreg complains that it gets no frequency ...
# trn = trn.asfreq('H')
# tst =tst.asfreq('H')


# trn = df.discharge[:trn_end].drop(columns=['Index'])
# tst = df.discharge[trn_end:].drop(columns=['Index'])

# Fit using statsmodel
l = np.array([1, 3, 4, 5])
mod = AutoReg(trn, l, old_names=False)
res = mod.fit()
print(res.summary())

data_ = res.predict(end=np.size(df.index) - 1, dynamic=trn_end)

# Assuming 'datetime' is the name of your datetime column
# df.set_index('datetime', inplace=True)
# df.index[0]
# df.index[-1]


fig, ax = plt.subplots()
ax.plot(df.t, df.discharge, "-", label="latent")
ax.plot(df.t[:trn_end], trn, "o", label="training")
ax.plot(df.t[trn_end:], tst, "x", label="testing")
ax.plot(data_.index, data_, "--", label="model")
plt.legend()

# # using integrated plot function
# fig = res.plot_predict(start=0, end=data.size-1)
# ax = plt.gca()
# ax.plot(t, data_latent, ":", label="latent")
# ax.plot(t[:trn_end], trn, "o", label="training")
# ax.plot(t[trn_end:], tst, "x", label="testing")
# plt.legend()

# plt.show()


# %%
train_cutoff = np.size(df) / 3
validate_cutoff = np.size(df) / 3 * 2

train_df = df[df['t'] <= train_cutoff]
select_df = df[(df['t'] > train_cutoff) & (df['t'] <= validate_cutoff)]
forecast_df = df[df['t'] > validate_cutoff]
# rain = pd.read_csv("/Users/ramunbar/Documents/Master/3_Semester/GITHUB/ETH_MP_Alpine_Water_Springs_Modelling/Data/meteo_data/resampled_precip_data/Freienbach/Freienbach_precip_H.csv")

# Assuming you have a DataFrame called 'df' with a column 'rain["rre150h0"]'

# Create a design matrix with 'rain["rre150h0"]' as an exogenous variable
# dm = dmatrix('rain["rre150h0"]-1', df)

# Optionally, add a constant term to the design matrix
# dm = add_constant(dm)

# Extract design matrices for different subsets of the data
# train_exog = build_design_matrices([dm.design_info], train_df, return_type='dataframe')[0]
# select_exog = build_design_matrices([dm.design_info], select_df, return_type='dataframe')[0]
# forecast_exog = build_design_matrices([dm.design_info], forecast_df, return_type='dataframe')[0]
# %%

plt.plot(train_df.t, train_df.discharge, label='Training data')
plt.plot(select_df.t, select_df.discharge, label='Model selection holdout')
plt.legend()
plt.title('Data overview')
plt.ylabel('discharge')
plt.xlabel('Index')
plt.show()

# %%
ar_model = AutoReg(endog=train_df.log_discharge, lags=5)
ar_fit = ar_model.fit()

train_log_pred = ar_fit.predict(start=train_df.t.min(), end=train_df.t.max())

plt.plot(train_df.t, train_df.discharge, label='Training data')
plt.plot(train_df.t,
         np.exp(train_log_pred), linestyle='dashed', label='In-sample prediction')
plt.legend()
plt.title('overview')
plt.ylabel('discharge')
plt.xlabel('hours')
plt.show()
# %%
select_log_pred = ar_fit.predict(start=select_df.t.min(), end=select_df.t.max())

plt.plot(train_df.t, train_df.discharge, label='Training data')
plt.plot(select_df.t, select_df.discharge, label='Model selection holdout')
plt.plot(train_df.t,
         np.exp(train_log_pred), linestyle='dashed', label='In-sample prediction')
plt.plot(select_df.t,
         np.exp(select_log_pred), linestyle='dashed', label='Validation set prediction')
plt.legend()
plt.title('overview')
plt.ylabel('discharge')
plt.xlabel('hours')
plt.show()

# %%

# split data
trn_end = 273
trn = df['nh4log'][:trn_end]
tst = df['nh4log'][trn_end:]

# Fit using statsmodel
mod = AutoReg(trn, 5, old_names=False)
res = mod.fit()
print(res.summary())

data_ = res.predict(end=df.size - 1, dynamic=trn_end)

t = np.arange(df.nh4.size)

fig, ax = plt.subplots()
ax.plot(t, df.nh4, "-", label="latent")
ax.plot(t[:trn_end], np.exp(trn), "o", label="training")
ax.plot(t[trn_end:], np.exp(tst), "x", label="testing")
ax.plot(t, np.exp(data_), "--", label="model")
plt.legend()