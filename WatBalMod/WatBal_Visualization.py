# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
import pickle  # copy figures
import matplotlib.pyplot as plt  # create plots
import plotly.graph_objects as go  # create interactive plots
from plotly.subplots import make_subplots
import os  # interaction with operating system
from datetime import datetime
import Helper


def first_impression_plot(spring_name, meteo_name, wb_df, variable_parameters):
    # Create a figure using Plotly graph objects
    fig = go.Figure()

    # Add the raw signal in blue
    fig.add_trace(
        go.Scatter(x=wb_df.index, y=wb_df['discharge_meas(mm)'] * variable_parameters['area'] / (60 * 24), mode='lines', name='discharge measured', line=dict(color='blue')))

    # Add smoothed signal in red
    fig.add_trace(go.Scatter(x=wb_df.index, y=wb_df['discharge_sim(mm)'] * variable_parameters['area'] / (60 * 24), mode='lines', name='discharge simulated',
                             line=dict(color='red')))

    # Customize the layout of the plot
    fig.update_xaxes(title_text='Datetime')
    fig.update_yaxes(title_text='Discharge (mm)')
    fig.update_layout(title=f'Spring {spring_name}, meteo {meteo_name}')
    fig.show()


def static_discharge_plot(spring_name, wb_df, variable_parameters):
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the raw signal in blue
    ax.plot(wb_df.index, wb_df['discharge_meas(mm)'] * variable_parameters['area'] / (60 * 24), linewidth=1, color='blue', label='discharge measured')

    # Plot the smoothed signal in orange
    ax.plot(wb_df.index, wb_df['discharge_sim(mm)'] * variable_parameters['area'] / (60 * 24), linewidth=1, color='orange', label='discharge simulated')

    # Customize the layout of the plot
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Discharge [L/min]')
    ax.set_title(f'{spring_name}')
    ax.legend()

    #ax.set_xlim([pd.to_datetime('2021-10-01'), pd.to_datetime('2022-07-20')])
    #ax.set_ylim([-5, 300])

    # Save the plot as a PDF
    current_time_str = datetime.now().strftime("%m_%d_%H")  # Convert the datetime object to a string
    save_path = os.path.join('Plots', 'WatBalMod', current_time_str)
    Helper.create_directory(save_path)
    fig.savefig(os.path.join(save_path, f'{spring_name}_model_.pdf'), bbox_inches='tight')
    plt.close(fig)
