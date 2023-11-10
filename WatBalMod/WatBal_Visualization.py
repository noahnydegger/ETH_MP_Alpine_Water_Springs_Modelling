# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
import pickle  # copy figures
import matplotlib.pyplot as plt  # create plots
import plotly.graph_objects as go  # create interactive plots
from plotly.subplots import make_subplots
import os  # interaction with operating system
import Helper


def first_impression_plot(spring_name, meteo_name, wb_df, catchment_parameters):
    # Create a figure using Plotly graph objects
    fig = go.Figure()

    # Add the raw signal in blue
    fig.add_trace(
        go.Scatter(x=wb_df.index, y=wb_df['discharge_meas(mm)'] *  catchment_parameters['area'] / (60 * 24), mode='lines', name='discharge measured', line=dict(color='blue')))

    # Add smoothed signal in red
    fig.add_trace(go.Scatter(x=wb_df.index, y=wb_df['discharge_sim(mm)'] *  catchment_parameters['area'] / (60 * 24), mode='lines', name='discharge simulated',
                             line=dict(color='red')))

    # Customize the layout of the plot
    fig.update_xaxes(title_text='Datetime')
    fig.update_yaxes(title_text='Discharge (mm)')
    fig.update_layout(title=f'Spring {spring_name}, meteo {meteo_name}')
    fig.show()
