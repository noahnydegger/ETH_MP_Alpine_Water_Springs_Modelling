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


def plot_Ulrika_static(spring_df, meteo_df, path_to_plot_folder, resolution='hourly', start=None, end=None):
    # Define color codes
    spring_c = 'blue'
    precip_c = 'midnightblue'

    # Convert start and end to datetime objects using pd.to_datetime
    start = pd.to_datetime(start) if start is not None else precip_df.index.min()
    end = pd.to_datetime(end) if end is not None else precip_df.index.max()
    # select subset of data
    spring_df = spring_df[pd.to_datetime(start):pd.to_datetime(end)]
    meteo_df = meteo_df[pd.to_datetime(start):pd.to_datetime(end)]

    # Create a figure
    fig, ax_flow = plt.subplots(figsize=(15, 9))

    # Plot the spring data
    ax_flow.plot(spring_df.index, spring_df['discharge(L/min)'], linewidth=1, color=spring_c, label='spring discharge')

    # Define aggregation frequencies using a dictionary
    aggregation_freqs = {'daily': 'D', 'monthly': 'M'}
    barwidths = {'daily': 1, 'monthly': 20}

    if resolution == 'hourly':
        precip_df = meteo_df
        barwidth = 1
    # Check if the selected resolution is in the dictionary
    elif resolution in aggregation_freqs:
        aggregation_freq = aggregation_freqs[resolution]
        barwidth = barwidths[resolution]
        # Aggregate the precipitation data
        precip_df = meteo_df['rre150h0'].resample(aggregation_freq).sum()
        precip_df = pd.DataFrame(precip_df, columns=['rre150h0'])
        precip_df.index = precip_df.index.to_period(aggregation_freq).to_timestamp(aggregation_freq)
    else:
        raise ValueError("Invalid resolution. Use 'hourly', 'daily', or 'monthly'.")

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


def plot_Ulrika_interactive(spring_df, meteo_df, resolution='hourly', start=None, end=None):
    from plotly.subplots import make_subplots

    # Convert start and end to datetime objects using pd.to_datetime
    start = pd.to_datetime(start) if start is not None else meteo_df.index.min()
    end = pd.to_datetime(end) if end is not None else meteo_df.index.max()

    # Select a subset of data within the specified date range
    spring_df = spring_df[pd.to_datetime(start):pd.to_datetime(end)]
    meteo_df = meteo_df[pd.to_datetime(start):pd.to_datetime(end)]

    # Define color codes
    spring_c = 'blue'
    precip_c = 'midnightblue'

    # Create an interactive figure using Plotly
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot the spring data on the primary y-axis
    fig.add_trace(go.Scatter(x=spring_df.index, y=spring_df['discharge(L/min)'], line=dict(width=1, color=spring_c), mode='lines', name='spring discharge'), secondary_y=False)

    if resolution == 'hourly':
        precip_df = meteo_df
    else:
        # Aggregate the precipitation data
        aggregation_freqs = {'daily': 'D', 'monthly': 'M'}
        if resolution in aggregation_freqs:
            aggregation_freq = aggregation_freqs[resolution]
            precip_df = meteo_df['rre150h0'].resample(aggregation_freq).sum()
            precip_df = pd.DataFrame(precip_df, columns=['rre150h0'])
            precip_df.index = precip_df.index.to_period(aggregation_freq).to_timestamp(aggregation_freq)
        else:
            raise ValueError("Invalid resolution. Use 'hourly', 'daily', or 'monthly'.")

    # Plot the precipitation data on the secondary y-axis
    fig.add_trace(go.Scatter(x=precip_df.index, y=precip_df['rre150h0'], marker=dict(color=precip_c, opacity=0.7), name=f'precipitation {resolution} sum', yaxis="y2"), secondary_y=True)

    # Configure the secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title=f'Precipitation {resolution} sum [mm/{resolution[0]}]',
            overlaying='y',
            side='right',
            autorange="reversed"  # Reverse the y-axis
        ),
    )

    # Invert the y-axis for precipitation
    #fig.update_yaxes(autorange="reversed")

    # Configure plot layout and labels
    fig.update_layout(
        title='Ulrika spring and Freienbach station',
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


def cross_correlation_time_series(series1, series2):
    # Ensure that both series have a date-time index
    if not isinstance(series1.index, pd.DatetimeIndex) or not isinstance(series2.index, pd.DatetimeIndex):
        raise ValueError("Both series must have a date-time index.")

    # Align the two series by the date-time index
    common_index = series1.index.intersection(series2.index)
    series1 = series1.reindex(common_index, fill_value=np.nan)
    series2 = series2.reindex(common_index, fill_value=np.nan)

    # Calculate the cross-correlation using numpy.correlate
    cross_corr = np.correlate(series1.values, series2.values, mode='full')

    # Compute the time lag corresponding to the maximum cross-correlation
    max_corr_index = np.argmax(cross_corr)
    time_lag = common_index[max_corr_index] - common_index[0]

    # Set a range for the plot around the maximum cross-correlation
    max_corr_range = 10  # Adjust this range as needed
    start_index = max_corr_index - max_corr_range
    end_index = max_corr_index + max_corr_range + 1

    # Create an array of time deltas corresponding to the cross-correlation values within the range
    time_deltas = pd.to_timedelta(np.arange(start_index, end_index), unit='H')

    # Extract the relevant portion of the cross-correlation
    cross_corr = cross_corr[start_index:end_index]

    # Plot the cross-correlation over the time deltas
    plt.figure(figsize=(10, 6))
    plt.plot(time_deltas, cross_corr, label=f'Time Lag: {time_lag}')
    plt.xlabel('Time Delta')
    plt.ylabel('Cross-Correlation')
    plt.title('Cross-Correlation of Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()

    return time_lag, cross_corr