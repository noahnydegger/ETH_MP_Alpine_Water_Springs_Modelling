# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
import pickle  # copy figures
import matplotlib.pyplot as plt  # create plots
import plotly.graph_objects as go  # create interactive plots
from plotly.subplots import make_subplots
import os  # interaction with operating system
from datetime import datetime


def first_impression_plot(spring_name, wb_df):
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the raw signal in blue
    ax.plot(wb_df.index, wb_df['discharge_meas(L/min)'], linewidth=1, color='blue', label='discharge measured')

    # Plot the smoothed signal in orange
    ax.plot(wb_df.index, wb_df['discharge_sim(L/min)'], linewidth=1, color='orange', label='discharge simulated')

    # Customize the layout of the plot
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Discharge [L/min]')
    ax.set_title(f'{spring_name}')
    ax.legend()

    #ax.set_xlim([pd.to_datetime('2021-10-01'), pd.to_datetime('2022-07-20')])
    #ax.set_ylim([-5, 300])
    plt.show()

    # Save the plot as a PDF
    current_time_str = datetime.now().strftime("%m_%d_%H")  # Convert the datetime object to a string
    #save_path = os.path.join('Plots', 'Water_Balance_Model', current_time_str)
    #Helper.create_directory(save_path)
    #fig.savefig(os.path.join(save_path, f'{spring_name}_model_.pdf'), bbox_inches='tight')

