# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
from scipy.stats import sem  # standard error
import pickle  # copy figures
import matplotlib.pyplot as plt  # create plots
import plotly.graph_objects as go  # create interactive plots
import csv  # read and write csv files
import os  # interaction with operating system


def plot_interactive_figure(spring_data_dfs, spring_names):
    # create interactive figure
    fig = go.Figure()
    for i in range(0, len(spring_data_dfs)):
        fig.add_trace(
            go.Scatter(x=spring_data_dfs[i].index, y=spring_data_dfs[i]['discharge(L/min)'], name=spring_names[i]))
    fig.update_layout(xaxis_title="Time", yaxis_title="discharge [L/min]")
    return fig


def plot_single_spring(spring_df, spring_description):
    # plot figure of standard errors
    fig, ax_prec = plt.subplots(figsize=(10, 6))  # create an empty figure
    ax_prec.plot(spring_df.index, spring_df['discharge(L/min)'], color='blue', label='discharge')
    ax_temp = ax_prec.twinx()
    ax_temp.plot(spring_df.index, spring_df['temperature(C)'], color='red', label='temperature')
    ax_prec.set_title(spring_description)
    ax_prec.set_ylabel('Discharge [L/min]', color='b')
    ax_temp.set_ylabel('Temperature [C]', color='r')
    ax_prec.set(xlabel='time', ylabel='discharge [L/min]')
    return fig
