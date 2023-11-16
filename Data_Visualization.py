# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
from scipy.stats import sem  # standard error
import pickle  # copy figures
import matplotlib.pyplot as plt  # create plots
import plotly.graph_objects as go  # create interactive plots
from plotly.subplots import make_subplots
import plotly.express as px
import csv  # read and write csv files
import os  # interaction with operating system
import Helper


def plot_interactive_figure(spring_data_dfs, spring_names):
    # create interactive figure
    fig = go.Figure()
    for i in range(0, len(spring_data_dfs)):
        fig.add_trace(
            go.Scatter(x=spring_data_dfs[i].index, y=spring_data_dfs[i]['discharge(L/min)'], name=spring_names[i]))
    fig.update_layout(xaxis_title="Time", yaxis_title="discharge [L/min]")
    return fig


def plot_single_spring(spring_name, spring_df, spring_description, path_to_plot_folder):
    try:
        # Create a figure
        fig, ax_flow = plt.subplots(figsize=(18, 12))

        # Plot data
        ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], color='blue', label='discharge',
                     linewidth=1.5)  # Increase linewidth
        ax_temp = ax_flow.twinx()
        ax_temp.plot(spring_df.index, spring_df['temperature(C)'], color='red', label='temperature',
                     linewidth=1.5)  # Increase linewidth

        # Customize the figure
        #ax_flow.set_title(spring_description)
        ax_flow.set_xlabel("", fontsize=18)  # Change the font size (16 in this example)
        ax_temp.set_ylabel('Air temperature [C]', color='r', fontsize=35)
        ax_flow.set_ylabel('Discharge [L/min]', color='b',fontsize=35)


        ax_flow.tick_params(axis='y', labelcolor='b',labelsize = "25")
        ax_temp.tick_params(axis='y', labelcolor='r',labelsize = "25")
        ax_flow.set(xlabel='Datetime')
        ax_flow.tick_params(axis='x', rotation=45, labelsize = "23")
        plt.grid(True)

        # Add legends
        #ax_flow.legend(loc='upper left')
        #ax_temp.legend(loc='upper right')
        #fig.tight_layout()

        # Save the plot as a PDF
        save_path = os.path.join(path_to_plot_folder, f"{spring_name}.pdf")
        fig.savefig(save_path)

        # Close the figure to free up resources
        plt.close(fig)

        return save_path  # Return the path where the figure is saved

    except Exception as e:
        # Handle any exceptions, e.g., print an error message
        print(f"An error occurred: {e}")
        return None  # Return None in case of an error


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def plot_snow(schnee_flims, schnee_thusis, figure_name):
    path_to_plot_folder = "/Users/ramunbar/Documents/Master/3_Semester/GITHUB/ETH_MP_Alpine_Water_Springs_Modelling/Plots/Meteo_Plots/Snow_Plots"

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Adjust the thickness of the bars
    bar_width = 0.5

    # Plot snow depth for schnee_flims as bars on the left y-axis
    ax1.bar(schnee_flims['measure_date'], schnee_flims['HS'], color='blue', label='Schnee Flims [cm]', width=bar_width,
            zorder=2)

    # Plot snow depth for schnee_thusis as bars on the left y-axis (in a different color)
    ax1.bar(schnee_thusis['measure_date'], schnee_thusis['HS'], color='green', label='Schnee Thusis [cm]',
            width=bar_width, zorder=3)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Snow Depth [cm]', color='blue')

    # Add horizontal grid lines
    ax1.yaxis.grid(True)

    # Create a second y-axis on the right for the change in snow depth
    # ax2 = ax1.twinx()
    # ax2.plot(schnee_flims['measure_date'], schnee_flims['delta HS'], color='red', label='Change in Snow Depth')
    # ax2.set_ylabel('Change in Snow Depth', color='red')
    ax1.tick_params(axis='x', rotation=45)

    # Show the legend
    ax1.legend(loc='upper left')

    # Save the figure with the specified name
    fig.savefig(os.path.join(path_to_plot_folder, '{}.pdf'.format(figure_name)))
    plt.close(fig)

def plot_snow_one(schnee_flims, figure_name):
        path_to_plot_folder = "/Users/ramunbar/Documents/Master/3_Semester/GITHUB/ETH_MP_Alpine_Water_Springs_Modelling/Plots/Meteo_Plots/Snow_Plots"

        # Create a figure and axis
        fig, ax1 = plt.subplots(figsize=(20, 6))

        # Adjust the thickness of the bars
        bar_width = 1

        # Plot snow depth for schnee_flims as bars on the left y-axis
        ax1.bar(schnee_flims['measure_date'], schnee_flims['HS'], color='blue', label='Schnee Flims [cm]',
                width=bar_width,
                zorder=2)



        ax1.set_xlabel('Year')
        ax1.set_ylabel('Snow Depth [cm]', color='blue')

        # Add horizontal grid lines
        ax1.yaxis.grid(True)

        # Create a second y-axis on the right for the change in snow depth
        # ax2 = ax1.twinx()
        # ax2.plot(schnee_flims['measure_date'], schnee_flims['delta HS'], color='red', label='Change in Snow Depth')
        # ax2.set_ylabel('Change in Snow Depth', color='red')
        #ax1.tick_params(axis='x', rotation=45)

        # Show the legend
        #ax1.legend(loc='upper left')

        # Save the figure with the specified name
        fig.savefig(os.path.join(path_to_plot_folder, '{}.pdf'.format(figure_name)))
        plt.close(fig)


import os
import matplotlib.pyplot as plt

def plot_spring_data_and_mc(spring_name, spring_df, spring_description, path_to_plot_folder, mc_data):
    try:
        # Create a figure
        fig, ax_flow = plt.subplots(figsize=(18, 12))

        # Plot data from spring_df
        ax_flow.plot(spring_df.index, spring_df["discharge(L/min)"], color='blue', label='discharge',
                     linewidth=1.5)  # Increase linewidth
        # Plot data from mc_data
        ax_flow.plot(mc_data["datetime"], mc_data["discharge [L/min]"], 'ro', markersize=10, label='MC Discharge')

        # Customize the figure
        ax_flow.set_xlabel("Datetime", fontsize=18)
        ax_flow.set_ylabel('Discharge [L/min]', color='b', fontsize=35)

        ax_flow.tick_params(axis='y', labelcolor='b', labelsize=25)
        ax_flow.set_xticklabels(spring_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S'), rotation=45, fontsize=23)
        plt.grid(True)

        # Add legend
        ax_flow.legend(loc='upper left')

        # Save the plot as a PDF
        save_path = os.path.join(path_to_plot_folder, f"{spring_name}.pdf")
        fig.savefig(save_path)
    except Exception as e:
        print(f"Error plotting {spring_name}: {str(e)}")

def plot_spring_precipitation_mc_interactive(mc_data, spring_name, meteo_name, resampled_spring_data_dfs, resampled_precip_data_dfs, resolution=('H', 'D'), start=None, end=None):

    # Define color codes
    spring_c = 'lightgreen'
    precip_c = 'blue'

    bar_widths = {'H': 0.2, 'D': 1, 'M': 1}  # width for the precipitation bars
    opacity_bar = {'H': 1, 'D': 1, 'M': 0.5}  # transparency for the precipitation bars

    # get temporal resolution for spring discharge and precipitation data
    res_spring = resolution[0]
    res_precip = resolution[1]

    spring_df = resampled_spring_data_dfs[spring_name][res_spring]

    if resampled_precip_data_dfs[meteo_name].get(res_precip) is None:
        res_precip = 'D'
    precip_df = resampled_precip_data_dfs[meteo_name][res_precip]

    # Convert start and end to datetime objects using pd.to_datetime
    start = pd.to_datetime(start, utc=True) if start is not None else precip_df.index.min()
    end = pd.to_datetime(end, utc=True) if end is not None else precip_df.index.max()

    # Select a subset of data within the specified date range
    spring_df = spring_df[start:end]
    precip_df = precip_df[start:end]

    # Create an interactive figure using Plotly
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot the spring data on the primary y-axis
    fig.add_trace(go.Scatter(x=spring_df.index, y=spring_df['discharge(L/min)'], line=dict(width=1, color=spring_c), mode='lines', name='spring discharge'), secondary_y=False)
    # Plot data from mc_data
    fig.add_trace(go.Scatter(x=mc_data["datetime"], y=mc_data['discharge [L/min]'], mode ='markers', marker=dict(size=8, color="red"),name='mc discharge'), secondary_y=False)

    # Plot the precipitation data on the secondary y-axis
    fig.add_trace(go.Bar(x=precip_df.index, y=precip_df['rre150h0'], opacity=opacity_bar[res_precip], marker=dict(color=precip_c, line=dict(color=precip_c, width=bar_widths[res_precip])), name=f'precipitation sum', yaxis="y2"), secondary_y=True)
    # Configure the secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title=f'Precipitation sum [mm/{res_precip.lower()}]',
            side='right',
            autorange="reversed"  # Reverse the y-axis
        ),
    )

    # Configure plot layout and labels
    fig.update_layout(
        title=f'{spring_name} spring ({res_spring}) and {meteo_name} Meteo station ({res_precip}) ',
        xaxis_title='Datetime',
        yaxis_title='Discharge [L/min]',
        xaxis=dict(tickangle=45),
        #showgrid=True
    )

    # Show the interactive plot
    fig.show()




def plot_meteo_precipitation(df, station, path_to_plot_folder):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data with a reversed y-axis
    ax.bar(df.index, df['rre150h0'])
    ax.invert_yaxis()  # Reverse the y-axis

    ax.set_xlabel('Datetime')
    ax.set_ylabel('Precipitation')
    ax.set_title(station)
    ax.grid(True)

    # Create masks for numeric and np.nan values
    numeric_mask = ~np.isnan(df['rre150h0'])
    nan_mask = np.isnan(df['rre150h0'])

    fixed_height = df['rre150h0'].max() / 2
    # Plot a line that is green where the data is numeric and red where it's np.nan
    ax.plot(df.index[numeric_mask], [fixed_height] * len(df.index[numeric_mask]), 'go', label='valid range', markersize=2)
    ax.plot(df.index[nan_mask], [fixed_height*1.01] * len(df.index[nan_mask]), 'ro', label='Nan range', markersize=2)

    # Set the x-axis range based on the minimum and maximum date values
    ax.set_xlim(df.index.min(), df.index.max())
    ax.tick_params(axis='x', rotation=45)

    # Add a legend to distinguish the lines
    ax.legend()

    fig.savefig(os.path.join(path_to_plot_folder, 'Meteo_Plots', '{}.pdf'.format(station)))
    plt.close(fig)

def plot_spring_mc_static(mc_data,spring_name, resampled_spring_data_dfs,
                                         save_path, name_extension, resolution=('H', 'D'), start=None, end=None):
        # Define color codes
        spring_c = 'blue'


        bar_widths = {'H': 0.2, 'D': 1, 'M': 20}  # width for the precipitation bars
        opacity_bar = {'H': 1, 'D': 0.8, 'M': 0.7}  # transparency for the precipitation bars

        # get temporal resolution for spring discharge and precipitation data
        res_spring = resolution[0]



        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        # Filter the DataFrame for the specified date range
        mc_data = mc_data[(mc_data['datetime'] >= start) & (mc_data['datetime'] <= end)]
        # select subset of data
        spring_df = resampled_spring_data_dfs[spring_name][res_spring][start:end]


        # Create a figure
        fig, ax_flow = plt.subplots(figsize=(15, 9))

        # Plot the spring data
        ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')

        # Plot the mc_data as red points
        ax_flow.scatter(mc_data['datetime'], mc_data['discharge [L/min]'], color='red', label='mc discharge', s=10)

        # Configure plot labels and titles

        ax_flow.set_ylabel('Discharge [L/min]', color=spring_c)

        ax_flow.tick_params(axis='y', labelcolor=spring_c)

        ax_flow.set(xlabel='Datetime')
        ax_flow.tick_params(axis='x', rotation=45)

        plt.grid(True)
        fig.tight_layout()

        # save the plot as a pdf
        save_path = os.path.join(save_path, spring_name)
        Helper.create_directory(save_path)
        fig.savefig(os.path.join(save_path, f'{spring_name}_{res_spring}_{name_extension}.pdf'))
        #plt.close(fig)
        print(mc_data)
        print(spring_df)

def plot_spring_precipitation_static(spring_name, meteo_names, resampled_spring_data_dfs, resampled_precip_data_dfs, save_path, name_extension, resolution=('H', 'D'), start=None, end=None):
    # Define color codes
    spring_c = 'blue'
    precip_c = 'midnightblue'

    bar_widths = {'H': 0.2, 'D': 1, 'M': 20}  # width for the precipitation bars
    opacity_bar = {'H': 1, 'D': 0.8, 'M': 0.7}  # transparency for the precipitation bars

    # get temporal resolution for spring discharge and precipitation data
    res_spring = resolution[0]
    res_precip = resolution[1]

    # Convert start and end to datetime objects using pd.to_datetime
    if resampled_precip_data_dfs[meteo_names[0]].get(res_precip) is None:
        res_precip = 'D'
    precip_df = resampled_precip_data_dfs[meteo_names[0]][res_precip]
    start = pd.to_datetime(start, utc=True) if start is not None else precip_df.index.min()
    end = pd.to_datetime(end, utc=True) if end is not None else precip_df.index.max()


    # select subset of data
    spring_df = resampled_spring_data_dfs[spring_name][res_spring][start:end]
    mc_data = mc_data
    nr_meteo = len(meteo_names)
    if nr_meteo > 1:  # more than one meteo station
        fig, axs = plt.subplots(nrows=nr_meteo + 1, ncols=1, figsize=(15, 9), sharex=True)  # create an empty figure
        # plot the spring data
        axs[nr_meteo].plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c,
                               label='spring discharge')
        axs[nr_meteo].set_title(f'{spring_name} spring at resolution {res_spring}')
        # create colored axis
        axs[nr_meteo].set_ylabel('Discharge [L/min]', color=spring_c)
        axs[nr_meteo].tick_params(axis='y', labelcolor=spring_c)
        # Plot the precipitation data with a reversed y-axis
        for i, meteo_name in enumerate(meteo_names):
            precip_df = resampled_precip_data_dfs[meteo_name][res_precip][start:end]
            axs[i].bar(precip_df.index, precip_df['rre150h0'], alpha=opacity_bar[res_precip], width=bar_widths[res_precip], color=precip_c, label='precipitation hourly sum')
            axs[i].invert_yaxis()  # Reverse the y-axis
            axs[i].set_title(f'{meteo_name} Meteo station at resolution {res_precip}')
            axs[i].tick_params(axis='y', labelcolor=precip_c)  # create colored axis

        axs[nr_meteo//2].set_ylabel(f'Precipitation sum [mm/{res_precip.lower()}]', color=precip_c)  # label only on second subplot

        # Set the x-axis range based on the minimum and maximum date values
        axs[nr_meteo].set_xlim(precip_df.index.min(), precip_df.index.max())
        axs[nr_meteo].set(xlabel='Datetime')
        axs[nr_meteo].tick_params(axis='x', rotation=45)
    else:  # only one meteo station data
        precip_df = resampled_precip_data_dfs[meteo_names[0]][res_precip][start:end]
        # Create a figure
        fig, ax_flow = plt.subplots(figsize=(15, 9))

        # Plot the spring data
        ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c,
                     label='spring discharge')

        # Create a secondary y-axis for precipitation
        ax_prec = ax_flow.twinx()
        ax_prec.bar(precip_df.index, precip_df['rre150h0'], color=precip_c, alpha=opacity_bar[res_precip], width=bar_widths[res_precip],
                    align='edge', label=f'precipitation {res_precip} sum')

        ax_prec.invert_yaxis()

        # Configure plot labels and titles
        ax_flow.set_title(f'{spring_name} spring ({res_spring}) and {meteo_names[0]} Meteo station ({res_precip})')
        ax_flow.set_ylabel('Discharge [L/min]', color=spring_c)
        ax_prec.set_ylabel(f'Precipitation sum [mm/{res_precip.lower()}]', color=precip_c)
        ax_flow.tick_params(axis='y', labelcolor=spring_c)
        ax_prec.tick_params(axis='y', labelcolor=precip_c)
        ax_flow.set(xlabel='Datetime')
        ax_flow.tick_params(axis='x', rotation=45)


    plt.grid(True)
    fig.tight_layout()

    # save the plot as a pdf
    save_path = os.path.join(save_path, spring_name)
    Helper.create_directory(save_path)
    fig.savefig(os.path.join(save_path, f'{spring_name}_{res_spring}_{res_precip}{name_extension}.pdf'))
    plt.close(fig)


def plot_spring_precipitation_interactive(spring_name, meteo_name, resampled_spring_data_dfs, resampled_precip_data_dfs, resolution=('H', 'D'), start=None, end=None):

    # Define color codes
    spring_c = 'lightgreen'
    precip_c = 'blue'

    bar_widths = {'H': 0.2, 'D': 1, 'M': 1}  # width for the precipitation bars
    opacity_bar = {'H': 1, 'D': 1, 'M': 0.5}  # transparency for the precipitation bars

    # get temporal resolution for spring discharge and precipitation data
    res_spring = resolution[0]
    res_precip = resolution[1]

    spring_df = resampled_spring_data_dfs[spring_name][res_spring]

    if resampled_precip_data_dfs[meteo_name].get(res_precip) is None:
        res_precip = 'D'
    precip_df = resampled_precip_data_dfs[meteo_name][res_precip]

    # Convert start and end to datetime objects using pd.to_datetime
    start = pd.to_datetime(start, utc=True) if start is not None else precip_df.index.min()
    end = pd.to_datetime(end, utc=True) if end is not None else precip_df.index.max()

    # Select a subset of data within the specified date range
    spring_df = spring_df[start:end]
    precip_df = precip_df[start:end]

    # Create an interactive figure using Plotly
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot the spring data on the primary y-axis
    fig.add_trace(go.Scatter(x=spring_df.index, y=spring_df['discharge(L/min)'], line=dict(width=1, color=spring_c), mode='lines', name='spring discharge'), secondary_y=False)

    # Plot the precipitation data on the secondary y-axis
    fig.add_trace(go.Bar(x=precip_df.index, y=precip_df['rre150h0'], opacity=opacity_bar[res_precip], marker=dict(color=precip_c, line=dict(color=precip_c, width=bar_widths[res_precip])), name=f'precipitation sum', yaxis="y2"), secondary_y=True)
    # Configure the secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title=f'Precipitation sum [mm/{res_precip.lower()}]',
            side='right',
            autorange="reversed"  # Reverse the y-axis
        ),
    )

    # Configure plot layout and labels
    fig.update_layout(
        title=f'{spring_name} spring ({res_spring}) and {meteo_name} Meteo station ({res_precip}) ',
        xaxis_title='Datetime',
        yaxis_title='Discharge [L/min]',
        xaxis=dict(tickangle=45),
        #showgrid=True
    )

    # Show the interactive plot
    fig.show()


def plot_spring_temperature_static(spring_name, meteo_names, resampled_spring_data_dfs, resampled_temp_data_dfs, save_path, name_extension, resolution=('H', 'D'), start=None, end=None):
    # Define color codes
    spring_c = 'blue'
    temp_c = ['r', 'g']
    window_size = {'H': 24, 'D': 10, 'W': 4, 'M': 1}
    lw = 2  # linewidth

    # get temporal resolution for spring discharge and precipitation data
    res_spring = resolution[0]
    res_temp = resolution[1]

    # Convert start and end to datetime objects using pd.to_datetime
    temp_df = resampled_temp_data_dfs[meteo_names[0]][res_temp]
    start = pd.to_datetime(start, utc=True) if start is not None else temp_df.index.min()
    end = pd.to_datetime(end, utc=True) if end is not None else temp_df.index.max()

    # select subset of data
    spring_df = resampled_spring_data_dfs[spring_name][res_spring]['temperature(C)'][start:end].rolling(window=7).mean()

    fig, ax = plt.subplots(figsize=(15, 9))  # create an empty figure
    # plot the spring data
    ax.plot(spring_df.index, spring_df.values, linewidth=lw, color=spring_c,
                           label=f'{spring_name} Water')

    # Plot the air temperature
    for i, meteo_name in enumerate(meteo_names):
        temp_df = resampled_temp_data_dfs[meteo_name][res_temp]['temperature(C)'][start:end].rolling(window=7).mean()
        ax.plot(temp_df.index, temp_df.values, linewidth=lw, color=temp_c[i], label=f'{meteo_name} Air')

    # Configure plot labels and titles
    ax.set_title(f'{spring_name} spring')
    ax.set_ylabel('Temperature [C]')
    ax.set_xlabel('Datetime')
    # Set the x-axis range based on the minimum and maximum date values
    ax.set_xlim(temp_df.index.min(), temp_df.index.max())
    ax.tick_params(axis='x', rotation=45)
    plt.grid(True)
    fig.tight_layout()
    # Add legends
    ax.legend(loc='upper center')

    # save the plot as a pdf
    save_path = os.path.join(save_path, spring_name)
    Helper.create_directory(save_path)
    fig.savefig(os.path.join(save_path, f'{spring_name}_{res_spring}_{res_temp}{name_extension}.pdf'))
    plt.close(fig)


def plot_spring_temperature_interactive(spring_name, meteo_names, resampled_spring_data_dfs, resampled_temp_data_dfs, resolution=('H', 'D'), start=None, end=None):
    # Define color codes
    spring_c = 'blue'
    temp_c = ['red', 'green']
    window_size = {'H': 24, 'D': 10, 'W': 4, 'M': 1}

    # get temporal resolution for spring discharge and precipitation data
    res_spring = resolution[0]
    res_temp = resolution[1]

    # Convert start and end to datetime objects using pd.to_datetime
    temp_df = resampled_temp_data_dfs[meteo_names[0]][res_temp]
    start = pd.to_datetime(start, utc=True) if start is not None else temp_df.index.min()
    end = pd.to_datetime(end, utc=True) if end is not None else temp_df.index.max()

    # select subset of data
    spring_df = resampled_spring_data_dfs[spring_name][res_spring]['temperature(C)'][start:end]

    # Create an interactive figure using Plotly
    fig = go.Figure()

    # Plot the spring data on the primary y-axis
    fig.add_trace(go.Scatter(x=spring_df.index, y=spring_df.values, line=dict(width=1, color=spring_c), mode='lines', name=spring_name))
    # Plot the air temperature
    for i, meteo_name in enumerate(meteo_names):
        temp_df = resampled_temp_data_dfs[meteo_name][res_temp]['temperature(C)'][start:end]
        temp_df_smoothed = temp_df.rolling(window=window_size[res_temp]).mean()
        # Plot the air temperature data on the secondary y-axis
        #fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df.values, line=dict(width=1, color=temp_c[i]), mode='lines', name=meteo_name))
        fig.add_trace(go.Scatter(x=temp_df_smoothed.index, y=temp_df_smoothed.values, line=dict(width=2, color=temp_c[i], dash='dash'), mode='lines',
                                 name=f'{meteo_name} smoothed'))

    # Configure plot layout and labels
    fig.update_layout(
        title=f'{spring_name} spring',
        xaxis_title='Datetime',
        yaxis_title='Temperature [C]',
        #showgrid=True
    )

    # Show the interactive plot
    fig.show()


def plot_cross_correlation_spring_meteo_multiple(spring_name, meteo_names, corr_dfs, range_of_days, save_path):
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r']
    # used to correctly display text on the plot
    corr_text = {}
    max_corr_order = {}

    for i, meteo_name in enumerate(meteo_names):
        time_lags_in_days = corr_dfs[meteo_name]['time_lag(days)']
        cross_corr = corr_dfs[meteo_name]['Pearson_corr']

        # You want to set the x-limits to -10 days and +10 days from the center
        center = len(time_lags_in_days) // 2  # Find the center index

        # Calculate the minimum and maximum x values
        x_center = time_lags_in_days[center]
        x_min = x_center - range_of_days
        x_max = x_center + range_of_days

        # Find the index of the overall maximum cross-correlation value
        max_corr_index = cross_corr.argmax()

        # Get the maximum cross-correlation value and its corresponding time delta
        max_corr_value = cross_corr[max_corr_index]
        max_time_delta = time_lags_in_days[max_corr_index]

        # Find the index of the maximum cross-correlation value within the specified range
        max_corr_index_in_range = (x_min <= time_lags_in_days) & (time_lags_in_days <= x_max)
        max_corr_value_in_range = cross_corr[max_corr_index_in_range].max()
        max_time_delta_in_range = time_lags_in_days[max_corr_index_in_range][cross_corr[max_corr_index_in_range].argmax()]

        # Calculate the position of the maximum cross-correlation value for the dashed line
        line_x = [max_time_delta_in_range, max_time_delta_in_range]
        line_y = [0, max_corr_value_in_range]

        # Plot the cross-correlation for the current dataset
        ax.plot(time_lags_in_days, cross_corr, color=colors[i], label=meteo_name)

        # Add a vertical dashed line at the position of the maximum cross-correlation value
        ax.plot(line_x, line_y, 'k--', linewidth=1)

        # to add the correlation value to the plot later
        max_corr_order[meteo_name] = (max_time_delta_in_range, max_corr_value_in_range)

        if max_corr_value_in_range == max_corr_value:  # the maximum correlation is within the specified range
            corr_text[meteo_name] = f'{meteo_name} Meteo station:\nMax Correlation: {max_corr_value:.2f}\n' \
                                    f'Time Delta: {max_time_delta:.2f} days'
        else:
            corr_text[meteo_name] = f'{meteo_name} Meteo station:\n' \
                                    f'Max Correlation in range: {max_corr_value_in_range:.2f}\n' \
                                    f'Time Delta in range: {max_time_delta_in_range:.2f} days\n' \
                                    f'Max Correlation overall:  {max_corr_value:.2f}\n' \
                                    f'Time Delta overall:  {max_time_delta:.2f} days'

    # Add the maximum cross-correlation value and time delta as text with background
    # Sort the dictionary based on the first element of the tuples in descending order
    max_corr_order = dict(sorted(max_corr_order.items(), key=lambda item: item[1][1], reverse=True))

    ha_pos = ['right', 'center', 'left']
    va_pos = ['top', 'center', 'top']
    h_offset = [-2.5, 0, 2.5]
    for i, meteo_name in enumerate(max_corr_order.keys()):
        x_pos = max_corr_order[meteo_name][0] + h_offset[i]
        y_pos = max_corr_order[meteo_name][1] * (1 + len(meteo_names) - i*1.5) / (1 + len(meteo_names))
        ax.text(x_pos, y_pos, corr_text[meteo_name], fontsize=12, ha=ha_pos[i], va=va_pos[i],
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))

    # plot settings
    ax.set_xlabel('Time Lag [d]')
    ax.set_ylabel('Pearson Cross-Correlation Value')
    ax.set_title(f'{spring_name} spring cross-correlation')
    ax.grid(True)

    # Set the x-limits to the calculated range
    ax.set_xlim(x_min, x_max)

    legend = ax.legend(title='Meteo station')
    legend.get_title().set_fontweight('bold')  # Set font weight

    # Save the plot as a PDF
    fig.savefig(os.path.join(save_path, f'{spring_name}_cross_correlation.pdf'), bbox_inches='tight')


def show_interactive_peak_plot(name, time_series, smoothed_signal, peaks):
    # Create a figure using Plotly graph objects
    fig = go.Figure()

    # Add the raw signal in blue
    fig.add_trace(
        go.Scatter(x=time_series.index, y=time_series.values, mode='lines', name='Raw Signal', line=dict(color='blue')))

    # Add smoothed signal in red
    fig.add_trace(go.Scatter(x=time_series.index, y=smoothed_signal, mode='lines', name='Smoothed Signal',
                             line=dict(color='red')))

    # Add the detected peaks on the smoothed signal
    fig.add_trace(go.Scatter(x=time_series.index[peaks], y=[smoothed_signal[i] for i in peaks], mode='markers',
                             name='Detected Peaks', marker=dict(size=8, color='green', symbol='x')))

    # Customize the layout of the plot
    fig.update_xaxes(title_text='Datetime')
    fig.update_yaxes(title_text='Signal Value')
    fig.update_layout(title=f'Peak Detection for spring {name}')
    fig.show()


def save_static_peak_plot(name, time_series, smoothed_signal, peaks, save_path):
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot the raw signal in blue
    ax.plot(time_series.index, time_series.values, linewidth=1, color='blue', label='Raw Signal')

    # Plot the smoothed signal in red
    ax.plot(time_series.index, smoothed_signal, linewidth=1, color='red', label='Smoothed Signal')

    # Scatter plot the detected peaks on the smoothed signal
    ax.scatter(time_series.index[peaks], [smoothed_signal[i] for i in peaks], color='green', marker='x', s=50,
               linewidths=2, label='Detected Peaks', zorder=2)

    # Customize the layout of the plot
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Discharge [L/min]')
    ax.set_title(f'Peak Detection for spring {name}')
    ax.legend()

    #ax.set_xlim([pd.to_datetime('2021-10-01'), pd.to_datetime('2022-07-20')])
    #ax.set_ylim([-5, 300])

    # Save the plot as a PDF
    fig.savefig(os.path.join(save_path, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_peak_width_boxplots(spring_peaks_dfs, save_path, show_labels=True):
    # Create a list to store the peak width data for each dataset
    peak_width_data = []
    selected_springs = ['Paliu_Fravi', 'Ulrika']

    # Extract the peak width data from each data frame and store it in the list
    for dataset_name, dataset_df in {key: spring_peaks_dfs[key] for key in selected_springs}.items():
        peak_width_data.append(dataset_df['Peak Width(h)'])

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()

    # Create a box plot for each dataset's peak width data
    #ax.boxplot(peak_width_data)

    # Create a box plot for each dataset's peak width data
    box = ax.boxplot(peak_width_data, showmeans=True)

    # Set the spring names as x-axis labels
    ax.set_xticklabels({key: spring_peaks_dfs[key] for key in selected_springs}.keys(), rotation=0)

    # Add numbers and labels on the plot
    if show_labels:
        fs = 12
        for i, vals in enumerate(peak_width_data):
            mean_val = np.mean(vals)
            ax.text(i + 1.1, mean_val, f'{mean_val:.1f}', ha='left', va='center', color='#519e3e', fontsize=fs)

            median_val = np.median(vals)
            ax.text(i + 1.1, median_val, f'{median_val:.1f}', ha='left', va='center', color='darkorange', fontsize=fs)

            # Add labels for lower and upper quartiles
            lower_q = np.percentile(vals, 25)
            upper_q = np.percentile(vals, 75)

            ax.text(i + 0.9, lower_q, f'{lower_q:.1f}', ha='right', va='center', color='black', fontsize=fs)
            ax.text(i + 0.9, upper_q, f'{upper_q:.1f}', ha='right', va='center', color='black', fontsize=fs)

        # Add labels for the box, mean, and median
        for i, box_line in enumerate(box['medians']):
            #ax.text(i + 1, box_line.get_ydata()[0], 'Median', ha='center', va='bottom', color='green')

            mean_x = i + 1.2
            mean_y = np.mean(box['means'][i].get_ydata())
            #ax.text(mean_x, mean_y, 'Mean', ha='left', va='center', color='blue')

    # Set labels and title
    ax.set_ylabel('Peak Width (hours)')
    #ax.set_xlabel('Springs')
    #ax.set_title('Peak Width Box Plots')

    # Show the plot
    plt.show()

    # Save the plot as a PDF
    Helper.create_directory(save_path)
    fig.savefig(os.path.join(save_path, 'spring_peaks_boxplots_selected_springs.pdf'), bbox_inches='tight')
