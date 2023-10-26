# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
import csv  # read and write csv files
import os  # interaction with operating system
from scipy.signal import savgol_filter, find_peaks, peak_widths, correlate

import Data_Visualization
import Helper


def cross_correlation_time_series_multipel(spring_name, meteo_names, resampled_spring_dfs, resampled_precip_dfs, path_to_plot_folder,resolution='H', range_of_days=10):
    # select spring dataframe
    spring_df = resampled_spring_dfs[spring_name][resolution]['discharge(L/min)']

    # Ensure that the spring_df series has a date-time index
    if not isinstance(spring_df.index, pd.DatetimeIndex):
        raise ValueError("The spring series must have a date-time index.")

    # Remove rows containing nan from the spring_df: otherwise correlate does not work
    spring_df.dropna(inplace=True)

    # Initialize dicts and lists to store time lags and Pearson correlations for each precip_df
    corr_dfs = {}
    time_lags_list = []
    pearson_corr_list = []

    for i, meteo_name in enumerate(meteo_names):
        # select meteo dataframe
        precip_df = resampled_precip_dfs[meteo_name][resolution]['rre150h0']
        # Ensure that the precip_df series has a date-time index
        if not isinstance(precip_df.index, pd.DatetimeIndex):
            raise ValueError("The precip_df series must have a date-time index.")

        # Remove rows containing nan from the precip_df: otherwise correlate does not work
        precip_df.dropna(inplace=True)

        # Align the spring and precip_df series by the date-time index
        common_index = spring_df.index.intersection(precip_df.index)
        spring_df_aligned = spring_df.loc[common_index]
        precip_df_aligned = precip_df.loc[common_index]

        # Calculate the cross-correlation using scipy.correlate
        cross_corr = correlate(spring_df_aligned.values, precip_df_aligned.values, mode='full')

        # Calculate the standard deviations
        std1 = np.std(spring_df_aligned)
        std2 = np.std(precip_df_aligned)

        # Calculate the Pearson correlation coefficient
        pearson_corr = cross_corr / (len(spring_df_aligned) * std1 * std2)

        # Extract the datetime index from the time series
        index = spring_df_aligned.index

        # Calculate the time lags based on the index
        time_lags_neg = index[0] - index
        time_lags_neg = time_lags_neg[::-1]
        time_lags_pos = index[1:] - index[0]
        time_lags = time_lags_neg.union(time_lags_pos)
        time_lags_in_days = time_lags.total_seconds() / (60 * 60 * 24)  # Convert to days

        # Store time lags and Pearson correlations as a dataframe in the dict
        corr_df = pd.DataFrame(index=time_lags)
        corr_df['time_lag(days)'] = time_lags_in_days
        corr_df['Pearson_corr'] = pearson_corr
        corr_df['cross_corr'] = cross_corr

        corr_dfs[meteo_name] = corr_df

    # Plot the cross-correlation for all precip_df
    save_path = os.path.join(path_to_plot_folder, 'spring_plots', 'spring_precip_correlation')
    Helper.create_directory(save_path)
    Data_Visualization.plot_cross_correlation_spring_precipitation_multiple(spring_name, meteo_names, corr_dfs, range_of_days, save_path)

    return corr_dfs

def find_spring_peaks(spring_name, discharge_df, path_to_plot_folder, window_length, polyorder, prominence_threshold, distance, show_plot=False, save_plot=False):
    # select the spring data
    discharge_df = discharge_df['discharge(L/min)']
    # Smooth the signal using Savitzky-Golay filter
    smoothed_signal = savgol_filter(discharge_df.values, window_length, polyorder)

    # Find peaks in the smoothed signal
    peaks, _ = find_peaks(smoothed_signal, prominence=prominence_threshold, distance=distance)

    # Calculate peak width using the smoothed signal
    widths, width_heights, left_ips, right_ips = peak_widths(smoothed_signal, peaks, rel_height=0.2)

    # Create a DataFrame for peak information
    peak_data = pd.DataFrame({'Datetime': discharge_df.index[peaks], 'Peak Value(L/min)': smoothed_signal[peaks], 'Peak Width(h)': widths/10})

    # Convert peak width to date range
    peak_data['Start Time'] = peak_data['Datetime'] - pd.to_timedelta(peak_data['Peak Width(h)'], unit='hours')
    peak_data['End Time'] = peak_data['Datetime'] + pd.to_timedelta(peak_data['Peak Width(h)'], unit='hours')

    if show_plot:
        Data_Visualization.show_interactive_peak_plot(spring_name, discharge_df, smoothed_signal, peaks)
    if save_plot:
        save_path = os.path.join(path_to_plot_folder, 'spring_plots', 'peak_detection')
        Helper.create_directory(save_path)
        Data_Visualization.save_static_peak_plot(spring_name, discharge_df, smoothed_signal, peaks, save_path)

    # Return the peak width values
    return peak_data


def spring_peaks_statistics(resampled_spring_data_dfs, spring_peaks_dfs):
    column_list = ['data duration', 'peak count', 'min width', '1st quartile', 'median width', 'mean width', '3rd quartile', 'max width']
    index_list = list(spring_peaks_dfs.keys())
    peak_statistics = pd.DataFrame(columns=column_list, index=spring_peaks_dfs.keys())
    stats = []
    for spring_name, peak_df in spring_peaks_dfs.items():
        if not peak_df.empty:
            spring_df = resampled_spring_data_dfs[spring_name]['10min']
            stats.append(spring_df.index[-1] - spring_df.index[0])  # data duration
            stats.append(peak_df['Peak Value(L/min)'].count())  # peak count
            stats.append(peak_df['Peak Width(h)'].min())  # min width
            stats.append(peak_df['Peak Width(h)'].quantile(q=0.25))  # 1st quartile
            stats.append(peak_df['Peak Width(h)'].quantile(q=0.5))  # median width
            stats.append(peak_df['Peak Width(h)'].mean())  # mean width
            stats.append(peak_df['Peak Width(h)'].quantile(q=0.75))  # d3rd quartile
            stats.append(peak_df['Peak Width(h)'].max())  # min width
        else:
            stats = [0] * len(column_list)

        peak_statistics.loc[spring_name] = stats
        stats = []

    return peak_statistics
