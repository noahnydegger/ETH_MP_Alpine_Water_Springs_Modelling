# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
from scipy.stats import sem  # standard error
import pickle  # copy figures
import matplotlib.pyplot as plt  # create plots
import plotly.graph_objects as go  # create interactive plots
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
    # plot figure
    fig, ax_flow = plt.subplots(figsize=(15, 9))  # create an empty figure
    ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], color='blue', label='discharge')
    ax_temp = ax_flow.twinx()
    ax_temp.plot(spring_df.index, spring_df['temperature(C)'], color='red', label='temperature')
    ax_flow.set_title(spring_description)
    # create colored axis
    ax_flow.set_ylabel('Discharge [L/min]', color='b')
    ax_temp.set_ylabel('Temperature [C]', color='r')
    ax_flow.tick_params(axis='y', labelcolor='b')
    ax_temp.tick_params(axis='y', labelcolor='r')
    ax_flow.set(xlabel='Datetime')
    ax_flow.tick_params(axis='x', rotation=45)
    plt.grid(True)
    # Add legends
    ax_flow.legend(loc='upper left')
    ax_temp.legend(loc='upper right')
    fig.tight_layout()

    # save the plot as a pdf
    fig.savefig(os.path.join(path_to_plot_folder, '{}.pdf'.format(spring_name)))
    plt.close(fig)
    return


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


def plot_Ulrika_static(spring_df, precip_df, path_to_plot_folder, resolution='hourly', start=None, end=None):
    # Define color codes
    spring_c = 'blue'
    precip_c = 'midnightblue'

    # Define aggregation frequencies using a dictionary
    aggregation_freqs = {'daily': 'D', 'monthly': 'M'}
    barwidths = {'daily': 1, 'monthly': 20}

    if resolution == 'hourly':
        precip_df = precip_df
        barwidth = 1
    # Check if the selected resolution is in the dictionary
    elif resolution in aggregation_freqs:
        aggregation_freq = aggregation_freqs[resolution]
        barwidth = barwidths[resolution]
        # Aggregate the precipitation data
        precip_df = precip_df['rre150h0'].resample(aggregation_freq).sum()
        precip_df = pd.DataFrame(precip_df, columns=['rre150h0'])
        precip_df.index = precip_df.index.to_period(aggregation_freq).to_timestamp(aggregation_freq)
    else:
        raise ValueError("Invalid resolution. Use 'hourly', 'daily', or 'monthly'.")

    # Convert start and end to datetime objects using pd.to_datetime
    start = pd.to_datetime(start) if start is not None else precip_df.index.min()
    end = pd.to_datetime(end) if end is not None else precip_df.index.max()
    # select subset of data
    spring_df = spring_df[pd.to_datetime(start):pd.to_datetime(end)]
    precip_df = precip_df[pd.to_datetime(start):pd.to_datetime(end)]

    # Create a figure
    fig, ax_flow = plt.subplots(figsize=(15, 9))

    # Plot the spring data
    ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')

    # Calculate the width based on the number of data points and the width of the x-axis
    barwidth = (precip_df.index[-1] - precip_df.index[0]) / len(precip_df)

    # Create a secondary y-axis for precipitation
    ax_prec = ax_flow.twinx()
    if resolution == 'hourly':
        ax_prec.bar(precip_df.index, precip_df['rre150h0'], color=precip_c, alpha=0.7,
                     label=f'precipitation {resolution} sum')
    else:
        ax_prec.bar(precip_df.index, precip_df['rre150h0'], color=precip_c, alpha=0.7, width=barwidth, align='edge',
                    label=f'precipitation {resolution} sum')

    ax_prec.invert_yaxis()

    # Set the x-axis range based on the provided start and end dates
    #ax_flow.set_xlim(start, end)
    ax_flow.tick_params(axis='x', rotation=45)

    # Configure plot labels and titles
    ax_flow.set_title('Ulrika spring and Freienbach station')
    ax_flow.set_ylabel('Discharge [L/min]', color=spring_c)
    ax_prec.set_ylabel(f'Precipitation {resolution} sum [mm/{resolution[0]}]', color=precip_c)
    ax_flow.tick_params(axis='y', labelcolor=spring_c)
    ax_prec.tick_params(axis='y', labelcolor=precip_c)
    ax_flow.set(xlabel='Datetime')
    ax_flow.tick_params(axis='x', rotation=45)
    plt.grid(True)

    fig.tight_layout()

    # Save the plot as a PDF
    fig.savefig(os.path.join(path_to_plot_folder, 'Discharge_and_Precipitation', f'Ulrika_{resolution}.pdf'))
    plt.close(fig)


def plot_spring_meteo_interactive(spring_df, precip_df, spring_name, meteo_station, resolution='hourly', start=None, end=None):
    from plotly.subplots import make_subplots

    if resolution == 'hourly':
        precip_df = precip_df
    else:
        # Aggregate the precipitation data
        aggregation_freqs = {'daily': 'D', 'monthly': 'M'}
        if resolution in aggregation_freqs:
            aggregation_freq = aggregation_freqs[resolution]
            precip_df = precip_df['rre150h0'].resample(aggregation_freq).sum()
            precip_df = pd.DataFrame(precip_df, columns=['rre150h0'])
            precip_df.index = precip_df.index.to_period(aggregation_freq).to_timestamp(aggregation_freq)
        else:
            raise ValueError("Invalid resolution. Use 'hourly', 'daily', or 'monthly'.")

    # Convert start and end to datetime objects using pd.to_datetime
    start = pd.to_datetime(start) if start is not None else precip_df.index.min()
    end = pd.to_datetime(end) if end is not None else precip_df.index.max()

    # Select a subset of data within the specified date range
    spring_df = spring_df[pd.to_datetime(start):pd.to_datetime(end)]
    precip_df = precip_df[pd.to_datetime(start):pd.to_datetime(end)]

    # Define color codes
    spring_c = 'lightgreen'
    precip_c = 'blue'

    # Create an interactive figure using Plotly
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot the spring data on the primary y-axis
    fig.add_trace(go.Scatter(x=spring_df.index, y=spring_df['discharge(L/min)'], line=dict(width=1, color=spring_c), mode='lines', name='spring discharge'), secondary_y=False)

    # Plot the precipitation data on the secondary y-axis
    #fig.add_trace(go.Scatter(x=precip_df.index, y=precip_df['rre150h0'], marker=dict(color=precip_c, opacity=0.7), spring_name=f'precipitation {resolution} sum', yaxis="y2"), secondary_y=True)
    fig.add_trace(go.Bar(x=precip_df.index, y=precip_df['rre150h0'], marker=dict(color=precip_c, line=dict(color=precip_c, width=1)), name=f'precipitation {resolution} sum', yaxis="y2"), secondary_y=True)
    # Configure the secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title=f'Precipitation {resolution} sum [mm/{resolution[0].lower()}]',
            #overlaying='y',
            side='right',
            autorange="reversed"  # Reverse the y-axis
        ),
    )

    # Invert the y-axis for precipitation
    #fig.update_yaxes(autorange="reversed")

    # Configure plot layout and labels
    fig.update_layout(
        title=f'{spring_name} spring and {meteo_station} station',
        xaxis_title='Datetime',
        yaxis_title='Discharge [L/min]',
        xaxis=dict(tickangle=45),
        #showgrid=True
    )

    # Show the interactive plot
    fig.show()


def plot_event_Ulrika(spring_df, meteo_df, path_to_plot_folder, start, end):
    spring_c = 'blue'
    precip_c = 'midnightblue'

    fig, ax_flow = plt.subplots(figsize=(15, 9))  # create an empty figure
    # plot the spring data
    ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')

    # Plot the precipitation data with a reversed y-axis
    ax_prec = ax_flow.twinx()
    ax_prec.bar(meteo_df.index, meteo_df['rre150h0'], width=0.03, color=precip_c, label='precipitation hourly sum')
    ax_prec.invert_yaxis()  # Reverse the y-axis

    # Set the x-axis range based on the minimum and maximum date values
    ax_flow.tick_params(axis='x', rotation=45)

    ax_flow.set_title('Ulrika spring and Freienbach station')
    # create colored axis
    ax_flow.set_ylabel('Discharge [L/min]', color=spring_c)
    ax_prec.set_ylabel('Precipitation hourly sum [mm/h]', color=precip_c)
    ax_flow.tick_params(axis='y', labelcolor=spring_c)
    ax_prec.tick_params(axis='y', labelcolor=precip_c)
    ax_flow.set(xlabel='Datetime')
    ax_flow.tick_params(axis='x', rotation=45)
    plt.grid(True)
    # Add legends
    #ax_flow.legend(loc='center left')
    #ax_prec.legend(loc='center right')
    fig.tight_layout()

    # save the plot as a pdf
    fig.savefig(os.path.join(path_to_plot_folder, 'Discharge_and_Precipitation', '{}_event.pdf'.format('Ulrika')))
    plt.close(fig)


def plot_monthly_Ulrika(spring_df, meteo_df, path_to_plot_folder):
    spring_c = 'blue'
    precip_c = 'midnightblue'
    fig, ax_flow = plt.subplots(figsize=(15, 9))  # create an empty figure
    # plot the spring data
    ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')

    # Plot the precipitation data with a reversed y-axis
    monthly_data = meteo_df['rre150h0'].resample('M').sum()
    # Create a DataFrame with the monthly sums
    monthly_df = pd.DataFrame(monthly_data, columns=['rre150h0'])
    monthly_df.index = monthly_data.index.to_period('M').to_timestamp('M')

    ax_prec = ax_flow.twinx()
    ax_prec.bar(monthly_df.index, monthly_df['rre150h0'], color=precip_c, alpha=0.7, width=20, align='edge',
                label='precipitation monthly sum')
    ax_prec.invert_yaxis()  # Reverse the y-axis

    # Set the x-axis range based on the minimum and maximum date values
    ax_flow.set_xlim(meteo_df.index.min(), meteo_df.index.max())
    ax_flow.tick_params(axis='x', rotation=45)

    ax_flow.set_title('Ulrika spring and Freienbach station')
    # create colored axis
    ax_flow.set_ylabel('Discharge [L/min]', color=spring_c)
    ax_prec.set_ylabel('Precipitation monthly sum [mm/h]', color=precip_c)
    ax_flow.tick_params(axis='y', labelcolor=spring_c)
    ax_prec.tick_params(axis='y', labelcolor=precip_c)
    ax_flow.set(xlabel='Datetime')
    ax_flow.tick_params(axis='x', rotation=45)
    plt.grid(True)
    # Add legends
    #ax_flow.legend(loc='center left')
    #ax_prec.legend(loc='center right')
    fig.tight_layout()

    # save the plot as a pdf
    fig.savefig(os.path.join(path_to_plot_folder, 'Discharge_and_Precipitation', '{}_month.pdf'.format('Ulrika')))
    plt.close(fig)


def plot_daily_Ulrika(spring_df, meteo_df, path_to_plot_folder):
    spring_c = 'blue'
    precip_c = 'midnightblue'
    fig, ax_flow = plt.subplots(figsize=(15, 9))  # create an empty figure
    # plot the spring data
    ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')

    # Plot the precipitation data with a reversed y-axis
    aggregated_data = meteo_df['rre150h0'].resample('D').sum()
    # Create a DataFrame with the monthly sums
    aggregated_df = pd.DataFrame(aggregated_data, columns=['rre150h0'])
    aggregated_df.index = aggregated_data.index.to_period('D').to_timestamp('D')

    ax_prec = ax_flow.twinx()
    ax_prec.bar(aggregated_df.index, aggregated_df['rre150h0'], color=precip_c, alpha=0.7, width=1, align='edge',
                label='precipitation daily sum')
    ax_prec.invert_yaxis()  # Reverse the y-axis

    # Set the x-axis range based on the minimum and maximum date values
    ax_flow.set_xlim(meteo_df.index.min(), meteo_df.index.max())
    ax_flow.tick_params(axis='x', rotation=45)

    ax_flow.set_title('Ulrika spring and Freienbach station')
    # create colored axis
    ax_flow.set_ylabel('Discharge [L/min]', color=spring_c)
    ax_prec.set_ylabel('Precipitation daily sum [mm/d]', color=precip_c)
    ax_flow.tick_params(axis='y', labelcolor=spring_c)
    ax_prec.tick_params(axis='y', labelcolor=precip_c)
    ax_flow.set(xlabel='Datetime')
    ax_flow.tick_params(axis='x', rotation=45)
    plt.grid(True)
    # Add legends
    #ax_flow.legend(loc='center left')
    #ax_prec.legend(loc='center right')
    fig.tight_layout()

    # save the plot as a pdf
    fig.savefig(os.path.join(path_to_plot_folder, 'Discharge_and_Precipitation', '{}_day.pdf'.format('Ulrika')))
    plt.close(fig)


def plot_daily_multiple_Ulrika(spring_df, meteo_dfs, path_to_plot_folder):
    spring_c = 'blue'
    precip_c = 'midnightblue'

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)  # create an empty figure
    # plot the spring data
    axs[-1].plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')
    axs[-1].set_title('Ulrika spring')
    # create colored axis
    axs[-1].set_ylabel('Discharge [L/min]', color=spring_c)
    axs[-1].tick_params(axis='y', labelcolor=spring_c)

    for i in range(0, 2):
        # Resample to get monthly sums
        meteo_df = meteo_dfs[i][1][1]
        aggregated_data = meteo_df['rre150h0'].resample('D').sum()
        # Create a DataFrame with the monthly sums
        aggregated_df = pd.DataFrame(aggregated_data, columns=['rre150h0'])
        aggregated_df.index = aggregated_data.index.to_period('D').to_timestamp('D')

        axs[i].bar(aggregated_df.index, aggregated_df['rre150h0'], color=precip_c, alpha=0.7, width=1, align='edge', label='precipitation monthly sum')
        axs[i].invert_yaxis()  # Reverse the y-axis
        axs[i].set_title(meteo_dfs[i][0])
        # create colored axis
        axs[i].tick_params(axis='y', labelcolor=precip_c)
        #position = axs[i].get_position()
        #axs[i].set_position([position.x0, position.y0, position.width, subplot_heights[i]])
    axs[1].set_ylabel('Precipitation daily sum [mm/day]', color=precip_c)

    # Set the x-axis range based on the minimum and maximum date values
    axs[-1].set_xlim(meteo_dfs[0][1][1].index.min(), meteo_dfs[0][1][1].index.max())
    axs[-1].set(xlabel='Datetime')
    axs[-1].tick_params(axis='x', rotation=45)

    plt.grid(True)
    # Add legends
    #ax_flow.legend(loc='center left')
    #ax_prec.legend(loc='center right')
    fig.tight_layout()

    # save the plot as a pdf
    fig.savefig(os.path.join(path_to_plot_folder, 'Discharge_and_Precipitation', '{}_day.pdf'.format('Ulrika_all')))
    plt.close(fig)

def plot_hourly_Paliu_Fravi(spring_df, meteo_dfs, path_to_plot_folder):
    spring_c = 'blue'
    precip_c = 'midnightblue'

    # Specify the relative heights of the subplots
    subplot_heights = [1, 1, 1, 2]  # The last subplot is twice as high

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 9), sharex=True)  # create an empty figure
    # plot the spring data
    axs[3].plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')
    axs[3].set_title('Paliu Fravi spring')
    # create colored axis
    axs[3].set_ylabel('Discharge [L/min]', color=spring_c)
    axs[3].tick_params(axis='y', labelcolor=spring_c)
    #position = axs[3].get_position()
    #axs[3].set_position([position.x0, position.y0, position.width, subplot_heights[0]])

    # Plot the precipitation data with a reversed y-axis
    #ax_prec = ax_flow.twinx()
    for i in range(0, 3):
        meteo_df = meteo_dfs[i][1][1]
        axs[i].bar(meteo_df.index, meteo_df['rre150h0'], color=precip_c, label='precipitation hourly sum')
        axs[i].invert_yaxis()  # Reverse the y-axis
        axs[i].set_title(meteo_dfs[i][0])
        # create colored axis
        axs[i].tick_params(axis='y', labelcolor=precip_c)
        #position = axs[i].get_position()
        #axs[i].set_position([position.x0, position.y0, position.width, subplot_heights[i]])
    axs[1].set_ylabel('Precipitation hourly sum [mm/h]', color=precip_c)

    # Set the x-axis range based on the minimum and maximum date values
    axs[3].set_xlim(meteo_dfs[0][1][1].index.min(), meteo_dfs[0][1][1].index.max())
    axs[3].set(xlabel='Datetime')
    axs[3].tick_params(axis='x', rotation=45)

    plt.grid(True)
    # Add legends
    #ax_flow.legend(loc='center left')
    #ax_prec.legend(loc='center right')
    fig.tight_layout()

    # save the plot as a pdf
    fig.savefig(os.path.join(path_to_plot_folder, 'Discharge_and_Precipitation', '{}_hour.pdf'.format('Paliu_Fravi_all')))
    plt.close(fig)


def plot_monthly_Paliu_Fravi(spring_df, meteo_dfs, path_to_plot_folder):
    spring_c = 'blue'
    precip_c = 'midnightblue'

    # Specify the relative heights of the subplots
    subplot_heights = [1, 1, 1, 2]  # The last subplot is twice as high

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(15, 9), sharex=True)  # create an empty figure
    # plot the spring data
    axs[3].plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')
    axs[3].set_title('Paliu Fravi spring')
    # create colored axis
    axs[3].set_ylabel('Discharge [L/min]', color=spring_c)
    axs[3].tick_params(axis='y', labelcolor=spring_c)
    #position = axs[3].get_position()
    #axs[3].set_position([position.x0, position.y0, position.width, subplot_heights[0]])

    # Plot the precipitation data with a reversed y-axis
    #ax_prec = ax_flow.twinx()
    for i in range(0, 3):
        # Resample to get monthly sums
        meteo_df = meteo_dfs[i][1][1]
        monthly_data = meteo_df['rre150h0'].resample('M').sum()
        # Create a DataFrame with the monthly sums
        monthly_df = pd.DataFrame(monthly_data, columns=['rre150h0'])
        monthly_df.index = monthly_data.index.to_period('M').to_timestamp('M')

        axs[i].bar(monthly_df.index, monthly_df['rre150h0'], color=precip_c, alpha=0.7, width=20, align='edge', label='precipitation monthly sum')
        axs[i].invert_yaxis()  # Reverse the y-axis
        axs[i].set_title(meteo_dfs[i][0])
        # create colored axis
        axs[i].tick_params(axis='y', labelcolor=precip_c)
        #position = axs[i].get_position()
        #axs[i].set_position([position.x0, position.y0, position.width, subplot_heights[i]])
    axs[1].set_ylabel('Precipitation monthly sum [mm/month]', color=precip_c)

    # Set the x-axis range based on the minimum and maximum date values
    axs[3].set_xlim(meteo_dfs[0][1][1].index.min(), meteo_dfs[0][1][1].index.max())
    axs[3].set(xlabel='Datetime')
    axs[3].tick_params(axis='x', rotation=45)

    plt.grid(True)
    # Add legends
    #ax_flow.legend(loc='center left')
    #ax_prec.legend(loc='center right')
    fig.tight_layout()

    # save the plot as a pdf
    fig.savefig(os.path.join(path_to_plot_folder, 'Discharge_and_Precipitation', '{}_month.pdf'.format('Paliu_Fravi_all')))
    plt.close(fig)


def plot_cross_correlation_spring_precipitation_single(spring_name, meteo_name, time_lags, cross_corr, range_of_days, save_path):
    # You want to set the x-limits to -10 days and +10 days from the center
    center = len(time_lags) // 2  # Find the center index
    time_lags_in_days = time_lags.total_seconds() / (60 * 60 * 24)  # Convert to days

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

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the cross-correlation
    ax.plot(time_lags_in_days, cross_corr, linewidth=2, label='Cross-Correlation')

    # Add a vertical dashed line at the position of the maximum cross-correlation value
    ax.plot(line_x, line_y, 'k--')

    # Add the maximum cross-correlation value and time delta as text with background
    if max_corr_value_in_range == max_corr_value:  # the maximum correlation is within the specified range
        ax.text(max_time_delta, max_corr_value * 0.5,
                f'Max Correlation: {max_corr_value:.2f}\nTime Delta: {max_time_delta:.2f} days', ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    else:
        ax.text(max_time_delta_in_range, max_corr_value_in_range * 0.5,
            f'Max Correlation in range: {max_corr_value_in_range:.2f}\n'
            f'Time Delta in range: {max_time_delta_in_range:.2f} days\n'
            f'Max Correlation overall:  {max_corr_value:.2f}\n'
            f'Time Delta overall:  {max_time_delta:.2f} days', ha='center', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    # plot settings
    ax.set_xlabel('Time Lag [d]')
    ax.set_ylabel('Cross-Correlation Value')
    ax.set_title(f'{spring_name} spring and {meteo_name} station cross-correlation')
    ax.grid(True)

    # Set the x-limits to the calculated range
    ax.set_xlim(x_min, x_max)

    ax.legend()

    # Save the plot as a PDF
    fig.savefig(os.path.join(save_path, f'{spring_name}_{meteo_name}_cross_correlation.pdf'), bbox_inches='tight')


def plot_cross_correlation_spring_precipitation_multiple(spring_name, meteo_names, corr_dfs, range_of_days, save_path):
    fig, ax = plt.subplots()

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
        ax.plot(time_lags_in_days, cross_corr, label=f'{meteo_name} station')

        # Add a vertical dashed line at the position of the maximum cross-correlation value
        ax.plot(line_x, line_y, 'k--')

        # Add the maximum cross-correlation value and time delta as text with background
        y_pos = max_corr_value_in_range * (i+1) / (len(meteo_name) + 1)
        if max_corr_value_in_range == max_corr_value:  # the maximum correlation is within the specified range
            ax.text(max_time_delta, max_corr_value * 0.5,
                    f'Max Correlation: {max_corr_value:.2f}\nTime Delta: {max_time_delta:.2f} days', ha='center', va='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        else:
            ax.text(max_time_delta_in_range, max_corr_value_in_range * 0.5,
                f'Max Correlation in range: {max_corr_value_in_range:.2f}\n'
                f'Time Delta in range: {max_time_delta_in_range:.2f} days\n'
                f'Max Correlation overall:  {max_corr_value:.2f}\n'
                f'Time Delta overall:  {max_time_delta:.2f} days', ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    # plot settings
    ax.set_xlabel('Time Lag [d]')
    ax.set_ylabel('Pearson Cross-Correlation Value')
    ax.set_title(f'{spring_name} spring cross-correlation')
    ax.grid(True)

    # Set the x-limits to the calculated range
    ax.set_xlim(x_min, x_max)

    ax.legend()

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
    fig, ax = plt.subplots()

    # Plot the raw signal in blue
    ax.plot(time_series.index, time_series.values, linewidth=1, color='blue', label='Raw Signal')

    # Plot the smoothed signal in red
    ax.plot(time_series.index, smoothed_signal, linewidth=1, color='red', label='Smoothed Signal')

    # Scatter plot the detected peaks on the smoothed signal
    ax.scatter(time_series.index[peaks], [smoothed_signal[i] for i in peaks], color='green', marker='x', s=100, linewidths=3, label='Detected Peaks')

    # Customize the layout of the plot
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Discharge [L/min]')
    ax.set_title(f'Peak Detection for spring {name}')
    ax.legend()

    # Save the plot as a PDF
    fig.savefig(os.path.join(save_path, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_peak_width_boxplots(spring_peaks_dfs, save_path):
    # Create a list to store the peak width data for each dataset
    peak_width_data = []

    # Extract the peak width data from each data frame and store it in the list
    for dataset_name, dataset_df in spring_peaks_dfs.items():
        peak_width_data.append(dataset_df['Peak Width(h)']/24)

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots()

    # Create a box plot for each dataset's peak width data
    ax.boxplot(peak_width_data)

    # Set the spring names as x-axis labels
    ax.set_xticklabels(spring_peaks_dfs.keys(), rotation=90)

    # Set labels and title
    ax.set_xlabel('Springs')
    ax.set_ylabel('Peak Width (days)')
    ax.set_title('Peak Width Box Plots')

    # Show the plot
    plt.show()

    # Save the plot as a PDF
    Helper.create_directory(save_path)
    fig.savefig(os.path.join(save_path, 'spring_peaks_boxplots.pdf'), bbox_inches='tight')
